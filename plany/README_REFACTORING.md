# Přehled plánu oprav a optimalizace: Aegis Engine & HUD

## 1. Cíl refaktoringu
Odstranit mikro-záseky a zamrzání uživatelského rozhraní (Pygame), eliminovat skryté CPU–GPU synchronizace (`cudaStreamSynchronize`) v simulační smyčce a vyčistit technický dluh v repozitáři bez narušení zpětné kompatibility uložených modelů (`.pt`) nebo determinismu simulace.

---

## 2. Přehled jednotlivých fází

| Fáze | Název a hlavní přínos | Dotčené soubory | Dokumentace | Priorita |
| :--- | :--- | :--- | :--- | :--- |
| **Fáze 1** | **Asynchronní ukládání snapshotů** (`Stage & Persist` přes `ThreadPoolExecutor` + atomický `os.replace`, konec zamrzání GUI při `F5` / `Ctrl+S`). | `src/tactical_hud.py`, `src/engine/async_runner.py`, `src/telemetry_logger.py` | `phase1_plan.md` | **Kritická** |
| **Fáze 2** | **Eliminace CPU–GPU synchronizací v horké smyčce** (odstranění re-entrantních zámků, vektorizace obsazování uzlů `(K, M)` na GPU a dávkový přenos telemetrie killů bez `.item()` ve smyčkách). | `src/engine_core.py` | `phase2_plan.md` | **Vysoká** |
| **Fáze 3** | **Lock-Free Double Buffering pro HUD** (rozdělení `_build_snapshot()` na krátký staging pod zámkem < 0,3 ms a čistě CPU formátování mimo zámek + atomická publikace reference). | `src/engine/async_runner.py`, `src/engine_core.py`, `src/tactical_hud.py` | `phase3_plan.md` | **Střední** |
| **Fáze 4** | **Čistota kódu a dokumentační konzistence** (oprava docstringu topologie sítě `14 -> 24 -> 16 -> 7` a odstranění 32 nepoužitých importů napříč 11 moduly). | `src/ai_brains.py` a dalších 10 souborů | `phase4_plan.md` | **Nízká (údržba)** |

---

## 3. Doporučené pořadí nasazení (Exekuční pipeline)

Pro minimalizaci rizika regresí a konfliktů při úpravách postupuj v tomto pořadí:

1. **Příprava prostředí:**
   * Vytvořit čistou pracovní větev: `git checkout -b refactor/async-io-and-gpu-sync`.
   * Spustit výchozí testovací sadu (`pytest tests/test_suite.py`) a zaznamenat referenční čas 1 000 tiků v headless režimu.
2. **Krok 1 – Fáze 4 (Úklid kódu a dokumentace):**
   * Rychlé promazání mrtvých importů (`F401`) a oprava docstringu bez zásahu do běhové logiky.
   * Ověření přes `ruff check . --select F401` a `pytest`.
3. **Krok 2 – Fáze 1 (Asynchronní snapshoty):**
   * Odstranění blokujícího zápisu na disk a `flush_spatial_parquet()` z hlavního vlákna Pygame.
   * Ověření plynulosti GUI při opakovaném stisku `F5` a zpětné kompatibility starších `.pt` snapshotů.
4. **Krok 3 – Fáze 3 (Lock-Free Double Buffering):**
   * Zkrácení doby držení `engine._engine_lock` při stavbě snapshotu pro HUD a přechod na atomickou výměnu reference v `SimulationBridge`.
   * Ověření konzistence zobrazovaných dat v HUDu při vysoké rychlosti simulace (`5x` / `10x`).
5. **Krok 4 – Fáze 2 (Vektorizace horké smyčky na GPU):**
   * Odstranění re-entrantních zámků v `_step_internal` a `_update_gates`, vektorizace obsazování uzlů a dávkový přenos killů.
   * Finální srovnání výkonu (čas na 1 000 tiků) vůči referenčnímu měření z Kroku 1.

---

## 4. Globální akceptační kritéria (Definition of Done)

- [ ] **Plynulost GUI:** Při 800 jednotkách na bojišti a ukládání snapshotu za běhu neklesne FPS v Pygame pod 55 snímků/s.
- [ ] **Zrychlení simulace:** Průměrný čas jednoho simulačního tiku (`_step_internal`) klesne minimálně o **20–30 %** oproti výchozímu stavu.
- [ ] **Stabilita vláken:** Při souběžném běhu simulace, HUDu a ukládání snapshotů nedochází k deadlockům ani k chybám typu `RuntimeError: deque/list mutated during iteration`.
- [ ] **Zpětná kompatibilita:** Starší uložené modely (`.pt`) lze bez problémů načíst přes GUI (`load_snap`) i přes `launcher.py`.
- [ ] **Čistý repozitář:** `pytest tests/test_suite.py` prochází na 100 % a statická analýza (`ruff` / `flake8`) hlásí `0 errors, 0 warnings`.

---

## 5. Navazující etapa (Po dokončení Fází 1–4)
* **Architektonická modularizace (`engine_core.py`):** Rozdělení monolitické třídy `RealisticBattleEngine` na samostatné, přehledné subsystémy (pohyb a fyzika, senzorika a rozhodování AI, ekonomika a evoluce epoch, správa stavu bitvy).

