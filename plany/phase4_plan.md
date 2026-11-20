# Implementační plán – Fáze 4: Údržba kódu, odstranění mrtvých importů a dokumentační konzistence

## 1. Shrnutí problému a ověřená diagnóza
* **Dotčené soubory:** 11 modulů napříč celým repozitářem (`launcher.py`, `run_headless.py`, `src/ai_brains.py`, `src/analyze_telemetry.py`, `src/engine/async_runner.py`, `src/engine/ballistics_system.py`, `src/engine/objectives_system.py`, `src/engine_core.py`, `src/tactical_hud.py`, `src/telemetry_lab.py`, `tests/test_suite.py`)
* **Identifikovaný technický dluh ze statické analýzy:**
  1. **Zastaralý docstring neuronové sítě (`src/ai_brains.py`, řádek 170):** Dokumentační komentář třídy genofondu stále uvádí starou topologii `(14 -> 64 -> 64 -> 7)`, zatímco reálné rozměry tenzorů v `__init__` (řádky 178–181) jsou `N_IN = 14`, `N_H1 = 24`, `N_H2 = 16`, `N_OUT = 7` (tedy topologie `14 -> 24 -> 16 -> 7`).
  2. **32 nepoužitých importů (`F401`) napříč 11 soubory:** V repozitáři zůstaly importy z dřívějších iterací vývoje (např. `math`, `np`, nevyužité konstanty tříd a technologií v `src/ai_brains.py` a `src/tactical_hud.py`, nebo nepoužité moduly v `run_headless.py` a `tests/test_suite.py`).

---

## 2. Cílová architektura (Čistý repozitář s `0 warnings` v linteru)

1. **Samodokumentující popis topologie v `src/ai_brains.py`:**
   * Opravit statický docstring na `(14 -> 24 -> 16 -> 7)` a doplnit vlastnost / helper `architecture_str` (např. `f"{self.N_IN} -> {self.N_H1} -> {self.N_H2} -> {self.N_OUT}"`), aby při budoucí změně šířky skrytých vrstev nedošlo znovu k rozchodu kódu a dokumentace.
2. **Bezpečné odstranění všech mrtvých importů (`F401`):**
   * Odstranit všechny identifikované nepoužité symboly a ověřit přes `ruff` / `pyflakes` a spuštění celé testovací sady, že žádný ze smazaných importů nebyl nepřímo re-exportován do jiného modulu (`from X import Y` přes prostředníka).

---

## 3. Detailní kroky implementace po souborech

### Krok 3.1: Oprava dokumentace architektury (`src/ai_brains.py`, řádky 170–187)

1. **Aktualizace docstringu na řádku 170:**
   * Změnit:
     `Třídně izolovaný hierarchický genofond (14 -> 64 -> 64 -> 7).`
   * Na:
     `Třídně izolovaný hierarchický genofond (14 -> 24 -> 16 -> 7).`
2. **Přidání dynamického atributu architektury v `__init__` (volitelné zpřesnění pro HUD/logy):**
   ```python
   self.architecture_str = f"{self.N_IN} -> {self.N_H1} -> {self.N_H2} -> {self.N_OUT}"
   ```

---

### Krok 3.2: Odstranění nepoužitých importů po jednotlivých souborech

*(Čísla řádků odpovídají stavu ze statické analýzy před Fází 1–3; při úpravě ověřit kontext řádku.)*

1. **`launcher.py`:**
   * Řádek 48: odstranit `datetime` (nebo zkontrolovat, zda se nepoužívá po úpravách Fáze 1).
2. **`run_headless.py`:**
   * Řádek 16: odstranit `datetime`
   * Řádek 17: odstranit `Path`
   * Řádek 36: odstranit `np` (`import numpy as np`)
   * Řádek 38: odstranit `scenarios`
   * Řádek 39: odstranit `ai_brains`
3. **`src/ai_brains.py`:**
   * Řádek 1: odstranit `import math`
   * Řádek 7: z importu konstant odstranit nepoužité symboly `CLASS_NAMES`, `TECH_NANITES`, `TECH_RADAR`, `TECH_SHIELDS`, `TECH_OVERCLOCK`, `TECH_AEROSOL`
4. **`src/analyze_telemetry.py`:**
   * Řádek 10: odstranit `import sys`
5. **`src/engine/async_runner.py`:**
   * Řádek 11: odstranit `Callable` z `typing` *(pouze pokud nebyl nově využit v typových anotacích callbacků z Fáze 1)*.
6. **`src/engine/ballistics_system.py`:**
   * Řádek 24: odstranit `import math`
   * Řádek 26: odstranit `import numpy as np`
7. **`src/engine/objectives_system.py`:**
   * Řádek 5: odstranit `import math`
8. **`src/engine_core.py`:**
   * Řádek 19: odstranit `import time`
   * Řádek 23: odstranit `worker_prof`
   * Řádek 25: odstranit `SCENARIO_CHASM`
9. **`src/tactical_hud.py`:**
   * Řádek 41: odstranit `FACTION_COLORS`
   * Řádek 42: z importu konstant odstranit `CLASS_GRUNT`, `CLASS_SNIPER`, `CLASS_JUGGERNAUT`, `TECH_NAMES`, `TECH_COSTS`
10. **`src/telemetry_lab.py`:**
    * Řádek 25: odstranit `import math`
11. **`tests/test_suite.py`:**
    * Řádek 18: odstranit `import numpy as np`
    * Řádek 29: odstranit `ai_brains`
    * Řádek 30: odstranit `scenarios`
    * Řádek 33: odstranit `telemetry_lab`

---

### Krok 3.3: Kontrola nepřímých re-exportů (Anti-regression check)

* Před commitem projet repozitář přes `grep` na symboly `SCENARIO_CHASM`, `FACTION_COLORS`, `CLASS_NAMES`, `TECH_NAMES` a `worker_prof`, aby se vyloučilo, že si je některý jiný modul importuje z `engine_core` nebo `tactical_hud` místo přímo ze zdrojového konfiguračního modulu.

---

## 4. Akceptační kritéria a ověřovací testy (Checklist)

- [ ] **Čistý linter (`0 errors, 0 warnings`):** Spuštění `ruff check . --select F401` (nebo `flake8 --select=F401`) v celém repozitáři vrátí **0 nalezených nepoužitých importů**.
- [ ] **Shoda dokumentace:** Docstring v `src/ai_brains.py` přesně odpovídá aktivní topologii sítě `(14 -> 24 -> 16 -> 7)`.
- [ ] **Plná průchodnost testů a vstupních bodů:**
  * `pytest tests/test_suite.py` projde bez `ImportError` nebo `NameError`.
  * Všechny tři hlavní vstupní body aplikace (`launcher.py`, `run_headless.py`, `src/analyze_telemetry.py`) se korektně spustí bez chyby importu.

