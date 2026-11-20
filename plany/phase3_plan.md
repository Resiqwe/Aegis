# Implementační plán – Fáze 3: Lock-Free Double Buffering pro telemetrii a HUD

## 1. Shrnutí problému a ověřená diagnóza
* **Dotčené soubory:** `src/engine/async_runner.py` (řádky 101–130), `src/engine_core.py` (metody `get_battle_state_snapshot` a `inspect_unit`)
* **Skutečná příčina blokování mezi simulačním workerem a GUI:**
  1. V metodě `AsyncSimulationWorker._build_snapshot()` (řádky 101–118) drží simulační vlákno zámek `self.engine._engine_lock` po celou dobu nejen kopírování GPU tenzorů, ale i následného čistě CPU formátování celého slovníku stavu bitvy (`get_battle_state_snapshot()`) a inspekce vybrané jednotky (`inspect_unit()`).
  2. Po celou tuto dobu nemůže hlavní GUI vlákno (Pygame) ani žádný příkaz z rozhraní získat přístup k enginu.
  3. Pokud `SimulationBridge` používá k předávání snapshotu do GUI sdílený mutovatelný slovník pod zámkem, GUI vlákno při každém překreslení (60+ FPS) soupeří o zámek s `Aegis-SimWorker` běžícím na 30 Hz.
* **Proč nepoužívat složité `pin_memory` buffery:**
  * Stav bojiště pro `N <= 800` jednotek má velikost pouze v řádu **desítek kilobajtů**. Jednorázové stažení přes `.detach().cpu().numpy()` (včetně mělkých/hlubokých kopií seznamů událostí) trvá v pauze mezi simulačními kroky pod **0,2 ms** a nevyžaduje realokaci bufferů při změně rozpočtu jednotek (`cycle_units`).

---

## 2. Cílová architektura (Stage → Format → Atomic Publish)

1. **Fáze 1: Staging (pod `engine._engine_lock`, doba běhu < 0,3 ms):**
   * Worker pod zámkem pouze vytáhne z enginu surové CPU kopie potřebných tenzorů (`.detach().cpu().numpy()`) a atomických metadat (epocha, čas, skóre, kopie seznamu `combat_events`, kopie částic ve `visual_mode`).
   * Rozhodne o `resolved_id` (pokud je aktivní `auto_best` nebo sledovaná jednotka zemřela) a vytáhne surová data inspekce jednotky.
   * Okamžitě uvolní `engine._engine_lock` – simulační engine je ihned volný pro další tik nebo GUI příkaz.
2. **Fáze 2: Format (mimo zámek v `Aegis-SimWorker`):**
   * Z připraveného surového CPU balíčku sestaví čistý, neměnný `snapshot` slovník pro HUD (bez jakéhokoli přístupu k `self.engine` nebo GPU).
3. **Fáze 3: Atomic Double-Buffer Publish (`SimulationBridge`):**
   * `SimulationBridge` drží referenci `self._latest_snapshot` (a počítadlo verzí `self._snapshot_seq`).
   * Publikace nového snímku je prosté atomické přiřazení reference (`self._latest_snapshot = snapshot`), které je v Pythonu díky GILu zaručeně atomické.
   * Pygame vlákno (`tactical_hud.py`) čte `bridge.get_latest_snapshot()` **zcela bez zámků (lock-free)** – dostane buď předchozí, nebo nový kompletní, konzistentní snímek.

---

## 3. Detailní kroky implementace po souborech

### Krok 3.1: Lock-free Double Buffer v `SimulationBridge` (`src/engine/async_runner.py`)

1. **Úprava ukládání a čtení snapshotu v `SimulationBridge`:**
   * Inicializovat v `__init__`:
     * `self._latest_snapshot: dict | None = None`
     * `self._snapshot_seq: int = 0`
   * Implementovat atomické metody bez zamykání (snapshot je po sestavení brán jako read-only / immutable):
     ```python
     def publish_snapshot(self, snapshot: dict) -> None:
         """Atomicky publikuje nový kompletní snapshot pro GUI vlákno (lock-free)."""
         self._snapshot_seq += 1
         snapshot["_seq"] = self._snapshot_seq
         self._latest_snapshot = snapshot

     def get_latest_snapshot(self) -> dict | None:
         """Vrátí poslední kompletní snapshot bez jakéhokoli zamykání."""
         return self._latest_snapshot
     ```
   * *(Poznámka: Výměna reference na objekt `self._latest_snapshot = snapshot` je v CPythonu atomická operace na úrovni bytecode `STORE_ATTR`. GUI vlákno nesmí vrácený slovník na místě mutovat.)*

---

### Krok 3.2: Rozdělení `_build_snapshot()` na Staging a Formatting (`src/engine/async_runner.py` & `src/engine_core.py`)

1. **Vytvoření metody `extract_raw_telemetry_state(self, target_id, auto_best) -> dict` v `src/engine_core.py`:**
   * Volá se uvnitř `with self._engine_lock:`.
   * Provede pouze rychlé zkopírování potřebných polí z GPU na CPU (`.detach().cpu().numpy()`) a zkopírování Python kolekcí (`list(self.combat_events)`, `list(self.particles)`), aby nedocházelo ke kolizi při dalším běhu simulace.
   * Vyhodnotí `resolved_id` (buď `target_id`, nebo `self.get_best_alive_unit()`) a vytáhne surové hodnoty pro inspekci dané jednotky (váhy/aktivace/stav).
   * Vrátí čistý slovník surových CPU polí `raw_state`.

2. **Vytvoření čistě CPU formátovacích funkcí `format_battle_snapshot(raw_state: dict) -> dict` a `format_inspected_unit(raw_state: dict) -> dict | None`:**
   * Pracují výhradně nad `raw_state` (CPU NumPy pole a Python skaláry) – **nulový přístup k `self.engine`, nulový přístup k GPU, nulové držení zámku**.
   * Původní metody `get_battle_state_snapshot()` a `inspect_unit()` na `RealisticBattleEngine` ponechat jako tenké synchronní obálky (`raw = self.extract_raw_telemetry_state(...); return format_battle_snapshot(raw)`) kvůli zpětné kompatibilitě s testy a single-thread režimem.

3. **Úprava `_build_snapshot(self) -> dict` v `src/engine/async_runner.py` (řádky 101–118):**
   ```python
   def _build_snapshot(self) -> dict:
       """
       2-fázové sestavení snapshotu:
       1. Rychlý staging pod _engine_lock (< 0.3 ms)
       2. Čistě CPU formátování slovníku mimo zámek
       """
       target_id, auto_best = self.bridge.get_selected_unit_request()

       # Fáze 1: Minimální kritická sekce pod zámkem
       with self.engine._engine_lock:
           raw_state = self.engine.extract_raw_telemetry_state(target_id, auto_best)

       # Fáze 2: CPU formátování mimo zámek (neblokuje engine ani GUI)
       snapshot = format_battle_snapshot(raw_state)
       inspected = format_inspected_unit(raw_state)
       snapshot["inspected_unit"] = inspected
       snapshot["selected_unit_id"] = inspected["unit_idx"] if inspected is not None else None
       return snapshot
   ```

4. **Publikace ve `_worker_loop`:**
   * V `_worker_loop()` po zavolání `snap = self._build_snapshot()` publikovat výsledek přes `self.bridge.publish_snapshot(snap)`.

---

## 4. Akceptační kritéria a ověřovací testy (Checklist)

- [ ] **Lock-free čtení v HUDu:** `tactical_hud.py` při vykreslování snímku nečte žádný sdílený slovník pod zámkem a nikdy nečeká na `engine._engine_lock`.
- [ ] **Minimální doba držení zámku:** Čas strávený uvnitř `with self.engine._engine_lock:` v `_build_snapshot()` nepřesáhne **0,5 ms** při 800 jednotkách.
- [ ] **Žádné částečné (torn) snímky:** HUD nikdy nevidí rozpracovaný stav (např. pozice jednotek z epochy N a skóre z epochy N+1) a nikdy nedojde k `RuntimeError: list/deque mutated during iteration` při vykreslování `combat_events` nebo `particles`.
- [ ] **Zpětná kompatibilita:** Původní metody `engine.get_battle_state_snapshot()` a `engine.inspect_unit()` nadále fungují v synchronním režimu i v `tests/test_suite.py`.

