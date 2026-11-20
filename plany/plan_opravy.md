# Manuál pro vypracování plánu oprav: Aegis Engine & HUD

## Cíl refaktoringu
Odstranit mikro-záseky a zamrzání uživatelského rozhraní (Pygame), eliminovat zbytečné CPU–GPU synchronizace v simulační smyčce a vyčistit technický dluh v repozitáři bez narušení zpětné kompatibility uložených modelů (`.pt`).

---

## Fáze 1: Ne-blokující ukládání snapshotů (Kritická priorita – GUI Freeze)

### Diagnóza problému
* **Soubory:** `src/tactical_hud.py` (řádky 283, 407), `src/telemetry_logger.py` (řádky 185–245)
* **Příčina:** GUI vlákno volá `save_snapshot()` synchronně. Metoda pod `engine._engine_lock` provádí 12 synchronních PCIe přenosů (`.cpu()`), zapisuje `.pt` soubor přes `torch.save()` (který drží Python GIL po celou dobu zápisu na disk) a spouští `flush_spatial_parquet()`. Po celou dobu stojí vykreslování i simulace.

### Požadovaná architektura řešení (2-fázový pipeline)
1. **Fáze Staging (pod `engine._engine_lock`, doba běhu < 1 ms):**
   * Zkopírovat tenzory z GPU do předalokovaných CPU bufferů s `pin_memory=True` pomocí `dst.copy_(src, non_blocking=True)`.
   * Zavolat jedinou synchronizaci `torch.cuda.synchronize()` (pokud běží na CUDA) a vytvořit mělkou kopii metadat (epochy, skóre, doktríny).
   * Okamžitě uvolnit `engine._engine_lock`.
2. **Fáze Persist (mimo zámek a mimo hlavní GIL):**
   * Předat připravená CPU data přes frontu do odděleného procesu (`multiprocessing.Process` nebo `concurrent.futures.ProcessPoolExecutor` s max 1 workerem), případně použít formát `safetensors` pro váhy + JSON pro metadata, aby zápis na disk neblokoval GIL vykreslovací smyčky.
   * `flush_spatial_parquet()` přesunout do stejného asynchronního I/O workeru.
3. **Úprava GUI (`src/tactical_hud.py`):**
   * Při stisku `F5` / `Ctrl+S` nebo kliknutí na tlačítko `save_snap` pouze zařadit požadavek (`worker.post_command("save_snapshot")`) a zobrazit banner *"Ukládání snapshotu zařazeno..."*. Potvrzení o uložení zobrazit až po dokončení zápisu.

### Akceptační kritéria Fáze 1
* Během ukládání snapshotu při běžící simulaci neklesne FPS v Pygame pod 55 snímků/s.
* Uložený soubor obsahuje pouze kompletní banky (`brains_W1_full` atd.) bez duplicitních oříznutých tenzorů `brains_W1[:N]` (zpětnou kompatibilitu zachovat pouze v `load_snapshot()`).

---

## Fáze 2: Eliminace skrytých CPU–GPU synchronizací v horké smyčce (Vysoká priorita – Výkon)

### Diagnóza problému
* **Soubor:** `src/engine_core.py` (řádky 504–519, 1200–1225, 1340–1365)
* **Příčina:**
  1. Dvojité zamykání v `step()` -> `_step_internal()`.
  2. Opakované volání `.item()` uvnitř `for` smyček (přes uzly a zabité jednotky), které vynucuje `cudaStreamSynchronize` na každý prvek.
  3. Podmínka `if in_node.any():` a `torch.where(alive)` v každém tiku vynucují synchronizaci GPU -> CPU.

### Požadovaná architektura řešení
1. **Odstranění duplicitního zámku:**
   * Odebrat `with self._engine_lock:` uvnitř `_step_internal()` a ponechat zámek pouze na veřejných vstupních bodech (`step()`, veřejné mutátory).
2. **Vektorizace obsazování uzlů (řádky 1202–1225):**
   * Nahradit Python smyčku `for i in range(num_nodes):` maticovým výpočtem vzdáleností všech živých jednotek ke všem uzlům najednou: tvar `(K, num_nodes)`.
   * Odstranit větev `if in_node.any():` – počítat přírůstky obsazování přes masky násobením nulou/jedničkou čistě na GPU.
3. **Dávkové stažení statistik killů (řádky 1344–1364):**
   * Místo volání `.item()` v cyklu stáhnout `k_t`, `last_attacker_id[k_t]`, `teams` a `pos[k_t]` jedním hromadným přenosem na CPU před smyčkou pro telemetrii, nebo akumulovat statistiky na GPU přes `index_add_` / `bincount`.

### Akceptační kritéria Fáze 2
* V metodě `_step_internal()` se nevyskytuje žádné volání `.item()` uvnitř `for` cyklu.
* Průměrný čas jednoho simulačního tiku (`_step_internal`) klesne minimálně o 20–30 % při plném počtu jednotek (800).

---

## Fáze 3: Lock-Free Double Buffering pro telemetrii a HUD (Střední priorita – Plynulost)

### Diagnóza problému
* **Soubor:** `src/engine/async_runner.py` (řádky 101–118)
* **Příčina:** `_build_snapshot()` drží `_engine_lock` po celou dobu sestavování kompletního slovníku pro HUD a inspekce jednotky.

### Požadovaná architektura řešení
1. **Minimální kritická sekce:**
   * Pod `_engine_lock` provést pouze kopii potřebných GPU tenzorů do `pinned` CPU bufferů.
   * Zámek uvolnit a veškeré formátování slovníků (`get_battle_state_snapshot`, `inspect_unit`) provést až na CPU mimo zámek.
2. **Atomický Double Buffer:**
   * Hotový snapshot publikovat atomickým přepsáním reference na sdíleném bridge objektu (`self.bridge.publish_snapshot(snapshot)`), aby GUI vlákno četlo vždy kompletní, neměnný objekt bez nutnosti zamykání.

### Akceptační kritéria Fáze 3
* Čtení snapshotu z Pygame vlákna nevyžaduje získání `engine._engine_lock`.

---

## Fáze 4: Údržba kódu a dokumentační konzistence (Nízká priorita – Čistota kódu)

### Úkoly k provedení
1. **Oprava docstringu (`src/ai_brains.py`, řádek 170):**
   * Aktualizovat popis architektury z `(14 -> 64 -> 64 -> 7)` na skutečnou topologii `(14 -> 24 -> 16 -> 7)` (nebo dynamicky generovat popis podle konstant `N_IN`, `N_H1`, `N_H2`, `N_OUT`).
2. **Pročištění nepoužitých importů:**
   * Odstranit mrtvé importy identifikované statickou analýzou v souborech:
     * `launcher.py` (`datetime`)
     * `run_headless.py` (`datetime`, `Path`, `np`, `scenarios`, `ai_brains`)
     * `src/ai_brains.py` (`math`, `CLASS_NAMES`, `TECH_NANITES`, `TECH_RADAR`, `TECH_SHIELDS`, `TECH_OVERCLOCK`, `TECH_AEROSOL`)
     * `src/analyze_telemetry.py` (`sys`)
     * `src/engine/async_runner.py` (`Callable`)
     * `src/engine/ballistics_system.py` (`math`, `np`)
     * `src/engine/objectives_system.py` (`math`)
     * `src/engine_core.py` (`time`, `worker_prof`, `SCENARIO_CHASM`)
     * `src/tactical_hud.py` (`FACTION_COLORS`, `CLASS_GRUNT`, `CLASS_SNIPER`, `CLASS_JUGGERNAUT`, `TECH_NAMES`, `TECH_COSTS`)
     * `src/telemetry_lab.py` (`math`)
     * `tests/test_suite.py` (`np`, `ai_brains`, `scenarios`, `telemetry_lab`)

### Akceptační kritéria Fáze 4
* Spuštění linteru (`ruff` / `flake8`) nevykazuje žádné chyby typu `F401` (unused imports).
* Všechny existující testy v `tests/test_suite.py` procházejí bez regrese.

---

## Doporučený postup nasazení
1. Vytvořit git větev `refactor/async-io-and-gpu-sync`.
2. Implementovat a otestovat **Fázi 4** (rychlé vyčištění bez změny logiky).
3. Implementovat **Fázi 1** a ověřit ukládání/načítání snapshotů za běhu.
4. Implementovat **Fázi 2** a **Fázi 3** a porovnat profilovací časy před a po úpravě.
