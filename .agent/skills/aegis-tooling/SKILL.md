---
name: aegis-tooling
description: Povinný diagnostický a kvótově úsporný protokol pro práci s projektem Cyber-Tactics 2.0 (Aegis Engine) pomocí lokálního nástroje scripts/aegis_tool.py.
---

# Aegis Tooling Protocol (Quota-Saving Workflow)

Při každé práci na projektu Cyber-Tactics 2.0 dodržuj tento postup, abys šetřil kontextové okno a pracoval s tvrdými, ověřenými daty místo slepého čtení tisíců řádků kódu:

## 1. Diagnostika před čtením souborů
Než začneš číst velké soubory (`src/engine_core.py`, `src/tactical_hud.py`, `launcher.py`), nejprve spusť příslušný lokální příkaz přes `.venv\Scripts\python.exe`:

* **Architektura, složitost, kotvy, GPU syncy, zámky:**
  ```powershell
  .venv\Scripts\python.exe scripts\aegis_tool.py xray --compact
  ```
  * X-Ray automaticky striktně hlídá nulové `core_hot_loop_sync` napříč všemi horkými moduly enginu (`src/engine_core.py` a celá `src/engine/*`).
  * Pro výpis souborů a řádků s tichými except bloky použij přepínač `--show-silent`:
    ```powershell
    .venv\Scripts\python.exe scripts\aegis_tool.py xray --compact --show-silent
    ```
* **Mapa čtení a zápisů atributů před modularizací třídy:**
  ```powershell
  .venv\Scripts\python.exe scripts\aegis_tool.py xray --deps RealisticBattleEngine
  ```
* **Sekční GPU/CPU profiling tiku (kde mizí milisekundy a VRAM alokace):**
  ```powershell
  .venv\Scripts\python.exe scripts\aegis_tool.py profile --compact
  ```
* **Inspekce uložených `.pt` modelů:**
  ```powershell
  .venv\Scripts\python.exe scripts\aegis_tool.py snaps list
  .venv\Scripts\python.exe scripts\aegis_tool.py snaps inspect <cesta.pt>
  ```

## 2. Cílené čtení řádků
Podle čísel řádků a sekčních kotev (`anchors`) z výstupu `xray` načti přes `read_file` **pouze konkrétní funkci nebo sekci** (typicky 40–100 řádků), nikdy zbytečně celý soubor.

## 3. Povinná regresní verifikace & Diff
* **Před každou výkonovou nebo logickou změnou** ulož baseline:
  ```powershell
  .venv\Scripts\python.exe scripts\aegis_tool.py bench --save-baseline
  ```
* **Po úpravě** ověř výkon, kontrakty tenzorů (`NaN`/`Inf`) a determinismus:
  ```powershell
  .venv\Scripts\python.exe scripts\aegis_tool.py bench --compare
  ```
* **Uložení diffu (úspora tokenů):**
  Místo výpisu stovek řádků do chatu ulož diff do souboru a zkontroluj pouze stručnou statistiku:
  ```powershell
  .venv\Scripts\python.exe scripts\aegis_tool.py diff <nazev_zmeny>
  ```
  Výsledný diff se automaticky uloží do `plany/<nazev_zmeny>.diff` a do konzole se vypíše pouze `git diff --stat`.
* **Zdravotní brána před commitem:**
  Kompletní 5fázová pre-commit brána (`py_compile` + `xray --strict` + `tests/test_suite.py` + `tests/test_golden_master.py` + `git status`):
  ```powershell
  .venv\Scripts\python.exe scripts\aegis_tool.py health
  ```
  Pro rychlý běh bez testu Golden Masteru lze použít přepínač `--fast`:
  ```powershell
  .venv\Scripts\python.exe scripts\aegis_tool.py health --fast
  ```

## 4. Batch Commits (No Auto-Commit)
Agenti po dílčích úpravách **necommitují automaticky** bez výslovného pokynu uživatele.
Po dokončení úpravy agent:
1. Uloží diff přes `aegis_tool.py diff <name>`,
2. Ověří stav přes `aegis_tool.py health`,
3. Ponechá ověřené změny v pracovní kopii (working tree) k hromadné revizi a commit provede až na explicitní žádost uživatele.
