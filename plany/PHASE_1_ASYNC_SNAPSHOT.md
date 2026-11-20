# Implementační plán – Fáze 1: Asynchronní ukládání snapshotů

## 1. Shrnutí problému a ověřená diagnóza
* **Dotčené soubory:** `src/tactical_hud.py`, `src/engine/async_runner.py`, `src/telemetry_logger.py`
* **Skutečná příčina zamrzání GUI:**
  1. Hlavní vykreslovací vlákno (Pygame) volá při stisku `F5` / `Ctrl+S` nebo kliknutí na tlačítko `save_snap` metodu `telemetry.save_snapshot(engine)` synchronně a napřímo.
  2. GUI vlákno tak uvízne na čekání na zámek `engine._engine_lock`, který drží simulační vlákno `Aegis-SimWorker` uprostřed výpočtu na GPU.
  3. Následné první volání `.cpu()` v `save_snapshot()` vynutí implicitní `torch.cuda.synchronize()` – GUI vlákno čeká na dokončení všech rozpracovaných CUDA kernelů a poté provádí blokující zápis `.pt` souboru i `flush_spatial_parquet()` přímo v hlavním vlákně.
* **Proč nepoužívat `multiprocessing` ani `pin_memory`:**
  * Celková velikost vah modelu (`800` slotů) je přibližně **2,8 MB**. Přenos do standardní CPU paměti přes `.cpu()` trvá v pauze mezi simulačními kroky méně než **0,3 ms**.
  * Použití `ProcessPoolExecutor` s aktivním CUDA kontextem přináší na Windows vysokou režii (`spawn`), riziko kolizí sdílené paměti a nutnost zbytečné meziprocesové komunikace.
  * Použití jednoho dedikovaného vlákna `ThreadPoolExecutor(max_workers=1)` pro diskové I/O v kombinaci se stagingem uvnitř `Aegis-SimWorker` problém plně a bezpečně vyřeší bez nových závislostí.

---

## 2. Cílová architektura (2-fázový pipeline: Stage & Persist)

1. **Spouštěč (GUI vlákno – `tactical_hud.py`):**
   * Nikdy nesahá na `engine._engine_lock` ani nevolá `telemetry.save_snapshot()` napřímo (pokud běží asynchronní worker).
   * Pouze vloží příkaz `"save_snapshot"` do `worker.post_command("save_snapshot")` a okamžitě pokračuje ve vykreslování.
2. **Fáze 1: Staging (`Aegis-SimWorker` – `async_runner.py` & `telemetry_logger.py`):**
   * Proběhne v `_process_commands()` mezi jednotlivými simulačními tiky, kdy GPU již dokončilo předchozí práci.
   * Pod `engine._engine_lock` zkopíruje GPU tenzory do CPU RAM pomocí `.detach().cpu().clone()` a vytvoří mělkou kopii skalárních metadat (hloubkovou kopii u měnitelných seznamů a `deque`).
   * Ukládá pouze kompletní genetické banky (`brains_W1_full` atd.) – duplicitní oříznuté kopie `brains_W1[:N]` se již neukládají.
   * Okamžitě uvolní `engine._engine_lock` (celková doba držení zámku < 1 ms).
3. **Fáze 2: Persist (`ThreadPoolExecutor(max_workers=1)` – `telemetry_logger.py`):**
   * Na pozadí mimo simulační i vykreslovací smyčku provede zápis připraveného CPU slovníku na disk (`torch.save` do dočasného souboru `.tmp` s atomickým přejmenováním `os.replace`, aby nikdy nevznikl poškozený soubor), provede `flush_spatial_parquet()` a zapíše záznam do `.jsonl` logu.
   * Po dokončení předá zpětnou notifikaci do HUDu pro zobrazení banneru s názvem uloženého souboru.

---

## 3. Detailní kroky implementace po souborech

### Krok 3.1: Úprava `src/telemetry_logger.py`

1. **Inicializace I/O exekutoru a stavu v `__init__`:**
   * Přidat `from concurrent.futures import ThreadPoolExecutor`.
   * V `__init__` inicializovat:
     * `self._io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Aegis-TelemetryIO")`
     * `self._save_in_progress = False`
     * `self._io_lock = threading.Lock()`

2. **Rozdělení `save_snapshot` na staging a zápis:**
   * Vytvořit interní metodu `_stage_snapshot_data(self, engine, custom_tag: str = "") -> tuple[str, str, dict]`:
     * Pod `with engine._engine_lock:` sestaví `filename`, `filepath` a slovník `snapshot_data`.
     * Všechny tenzory převede přes `.detach().cpu().clone()`.
     * U kolekcí `history_best` a `history_avg` (typu `deque`) vytvoří kopii (např. `[ deque(d, maxlen=d.maxlen) for d in engine.brains.history_best_fitness ]`), aby nedošlo k chybě `RuntimeError: deque mutated during iteration` při souběžném běhu další epochy.
     * Odstraní duplicitní klíče `brains_W1`, `brains_b1`, `brains_W2`, `brains_b2`.
   * Vytvořit interní metodu `_persist_snapshot_to_disk(self, filepath: str, filename: str, snapshot_data: dict) -> str`:
     * Zapíše data do `tmp_path = filepath + ".tmp"` pomocí `torch.save(snapshot_data, tmp_path)` a následně provede `os.replace(tmp_path, filepath)`.
     * Zavolá `self.flush_spatial_parquet()` (ověřit, že čtení bufferu pro Parquet je thread-safe – případně také vykopírovat seznam záznamů ve staging fázi).
     * Zapíše `SNAPSHOT_SAVED` záznam do `self.log_filepath`.
     * V bloku `finally:` shodí příznak `self._save_in_progress = False`.

3. **Přidání veřejné metody `save_snapshot_async(self, engine, custom_tag: str = "", on_complete=None) -> bool`:**
   * Zkontroluje pod `self._io_lock`, zda `self._save_in_progress` není `True`. Pokud ano, vrátí `False` (ochrana proti zběsilému mačkání `F5`).
   * Nastaví `self._save_in_progress = True`.
   * Zavolá `filepath, filename, data = self._stage_snapshot_data(engine, custom_tag)`.
   * Odešle úlohu do `self._io_executor.submit(self._persist_snapshot_to_disk, filepath, filename, data)`.
   * Pokud je předán callback `on_complete`, naváže ho přes `future.add_done_callback(...)`.
   * Původní synchronní `save_snapshot()` ponechat jako wrapper (zavolá `_stage_snapshot_data` a hned `_persist_snapshot_to_disk`) pro případ headless režimu nebo testů.

4. **Zpětná kompatibilita v `load_snapshot(self, engine, filepath: str) -> bool`:**
   * Ověřit, že načítání preferuje klíče `brains_W1_full`, `brains_b1_full`, `brains_W2_full`, `brains_b2_full`, `brains_W3_full`, `brains_b3_full`.
   * Pokud v načítaném starším souboru `_full` klíče chybí, použít fallback na staré klíče `brains_W1` atd.

5. **Korektní ukončení (`shutdown` / `close`):**
   * Přidat metodu `shutdown(self, wait: bool = True)`, která zavolá `self._io_executor.shutdown(wait=wait)`, aby se při vypnutí hry bezpečně dopsal rozpracovaný snapshot.

---

### Krok 3.2: Úprava `src/engine/async_runner.py`

1. **Rozšíření komunikačního bridge (notifikace do GUI):**
   * Ověřit (případně doplnit do `SimulationBridge`), jednoduchou thread-safe frontu zpráv pro HUD: `post_banner(msg: str)` a `pop_banners() -> list[str]`.
2. **Obsluha příkazu v `_process_commands(self)`:**
   * Přidat větev:
     ```python
     elif cmd_name == "save_snapshot":
         if self.engine.telemetry_mgr is not None:
             started = self.engine.telemetry_mgr.save_snapshot_async(
                 self.engine,
                 *args,
                 on_complete=lambda path: self.bridge.post_banner(
                     f"[ULOZIT] Uložen model: {os.path.basename(path)}"
                 ),
             )
             if not started:
                 self.bridge.post_banner("[INFO] Ukládání snapshotu již probíhá...")
     ```
3. **Napojení na `stop()`:**
   * Při zastavení workeru (`stop()`) zajistit po ukončení smyčky volání `self.engine.telemetry_mgr.shutdown(wait=True)` (pokud se vypíná celá aplikace).

---

### Krok 3.3: Úprava `src/tactical_hud.py`

1. **Náhrada přímého volání u kláves `F5` / `Ctrl+S` (kolem řádku 282):**
   * Pokud běží `self.worker`, volat `self.worker.post_command("save_snapshot")` a zobrazit krátký banner `"Ukládání snapshotu na pozadí..."` (1.5 s).
   * Fallback na synchronní `self.telemetry.save_snapshot(self.engine)` ponechat pouze pro případ, kdy `self.worker is None`.
2. **Náhrada přímého volání u tlačítka `save_snap` (kolem řádku 406):**
   * Stejná logika přes `self.worker.post_command("save_snapshot")`.
3. **Vykreslení potvrzovacího banneru:**
   * V hlavní smyčce HUDu (v každém snímku) zkontrolovat `self.bridge.pop_banners()` a pro každou zprávu zavolat `self.show_banner(msg, 3.0)`.

---

## 4. Akceptační kritéria a ověřovací testy (Checklist)

- [ ] **Plynulost GUI:** Při běhu 800 jednotek a opakovaném stisku `F5` nedochází k zamrznutí okna ani viditelnému propadu FPS.
- [ ] **Odolnost proti spamu (Debounce):** Rychlé 5násobné stisknutí `F5` za sebou nespustí 5 souběžných zápisů ani nevyhodí výjimku; zobrazí se informace o probíhajícím ukládání.
- [ ] **Thread-safety kolekcí:** Během ukládání za plného běhu (rychlost 5x/10x, rychlé střídání epoch) nevzniká `RuntimeError: deque mutated during iteration` ani souběžný přístup k Parquet bufferu.
- [ ] **Atomický zápis:** Během ukládání se nejprve tvoří `.pt.tmp` a teprve po dokončení se přejmenuje na `.pt` (žádné neúplné soubory při pádu aplikace).
- [ ] **Zpětná kompatibilita:** Nově uložený snapshot (bez duplicitních oříznutých tenzorů) jde načíst přes `load_snap` v HUDu i přes `launcher.py`, a zároveň stále jdou načíst starší snapshoty.

