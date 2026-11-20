# Oprava race condition (synchronizace vláken) mezi simulačním workerem a GUI vláknem

## Diagnóza problému
Při přechodu epochy nebo kliknutí do GUI docházelo ke kolizi vláken, protože GUI vlákno čte živé GPU tenzory (pos, hp, brains) mimo zámek `self._engine_lock` (threading.RLock()), zatímco worker vlákno je v `start_new_operation` nebo `finish_operation_and_evolve` přepisuje.

## Úprava č. 1: `src/engine_core.py` – metoda `inspect_unit`

**Co:** Celé tělo metody `inspect_unit` (řádky 1518–1573) obaleno do `with self._engine_lock:`.

**Proč:** Metoda čte živé GPU tenzory (`pos`, `hp`, `brains`, `classes`, `teams`, `last_sensors`, `last_target_enemy` atd.) bez ochrany zámekem. Při kliknutí myší nebo událostech GUI docházelo k kolizi vláken, kdy worker vlákno přepisuje data v `start_new_operation` nebo `finish_operation_and_evolve` zatímco GUI čte data ve stavu přechodu epochy.

**Důležité:** Používáno `threading.RLock()` (reentrantí zámek), což umožňuje vnořené volání metod bez deadlocku. Zachovává stávající výpočetní logiku – pouze přidává ochranu přístupu k tenzorům.

---

## Úprava č. 2: `launcher.py` – tři změny

### a) `_draw_modern_dock` (řádky 605–644)
- Přidal volitelný parameter `state: dict | None = None`
- Když `state` není None (typicky z `get_battle_state_snapshot()`), čte `alive_counts`, `initial_team_counts`, `generals` z tohoto snapshotu místo přímého přístupu k `self.engine.hp`, `self.engine.N`, `self.engine.current_epoch`
- Při `state = None` (backup případ) zachovává původní chování

### b) `render` (řádky 489–490)
- Změna volání: `_draw_modern_dock(dock_rect)` → `_draw_modern_dock(dock_rect, state)`
- Tím se zajistí, že dock panel používí synchronizovaný snapshot namísto přímého čtení engine tenzorů

### c) `_draw_modern_analytics` (řádky 832–900)
- Původní: `hist = getattr(self.engine.brains, "history_avg_fitness", ...)` – přímé čtení GPU tenzoru
- Opraveno: `hist = state.get("history_avg_fitness", ...)` – čtení z CPU snapshotu, který je už zabezpečen zámekem ve `get_battle_state_snapshot()`

---

## Úprava č. 3: `src/tactical_hud.py` – metoda `handle_events` (řádky 303–318)

**Co:** Přístup k `self.engine.pos.cpu().numpy()` obalen `with self.engine._engine_lock:`.

**Kód před:** `pos_np = self.engine.pos.cpu().numpy()`
**Kód po:** `with self.engine._engine_lock: pos_np = self.engine.pos.cpu().numpy()`

**Proč:** V sekci `MOUSEBUTTONDOWN` (kliknutí myší) docházelo k race condition, kdy worker vlákno mohlo v tu chvíli měnit pozice vojáků (provádí se v `_step_internal`). Úprava zajišťuje, že čtení pozic probíhá pod zámekem.

**Bezpečnost:** Používá `RLock` (stejného typu jako v engine_core.py), takže nedojde k deadlocku při vnořených voláních metod.

---

## Shrnutí principu řešení

Všechny úpravy řídí stejný princip:

1. **GUI vlákno** nyní čte data buď:
   - Ze *snapshotu v RAM* (CPU pole, která jsou již produkována workerem pod `_engine_lock`), nebo
   - Přímě pod `_engine_lock` (přechodná ochrana)

2. **Worker vlákno** vždy drží `_engine_lock` při zápise do GPU tenzorů (metody `start_new_operation`, `finish_operation_and_evolve`, `_step_internal`, `get_battle_state_snapshot` všechny používají `with self._engine_lock:`)

3. **Žádná změna herní/logiky:** Tensorová matematika, struktura dat, názvy metod – liche.

4. **Použití `RLock` namísto `Lock`:** Ujistilo se, že žádný deadlock nevznikne při vnořených voláních (např. HUD → engine → worker, kdy může být lock již držen).

Všechny změny byly ověněny syntaxí Pythonu a konzistence použití `_engine_lock` napříč celým kódem.