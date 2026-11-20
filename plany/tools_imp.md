# Finální implementační manuál: Aegis Tooling Kit (`scripts/aegis_tool.py`)

> **Účel dokumentu:** Finální zadání pro autonomního vývojového agenta po křížovém architektonickém auditu.
> **Cílový projekt:** Cyber-Tactics 2.0 / Aegis Engine (PyTorch/CUDA neuroevoluční simulátor, Pygame HUD, Python 3.12, Windows 11, RTX 5070 Ti).
> **Stav repozitáře před implementací:** Fáze 1 (async snapshoty), Fáze 3 (lock-free double buffer) a Fáze 4 (úklid kódu a git historie) jsou dokončeny; Fáze 2 (vektorizace GPU smyčky) je zavedena.

---

## 1. Proč tento nástroj stavíme (Kontext & Motivace pro agenta)

1. **Úspora kontextového okna (80–90 % méně tokenů):** Místo opakovaného čtení celých 1 000+ řádkových souborů (`src/engine_core.py`, `src/tactical_hud.py`, `launcher.py`) agent nejprve spustí lokální AST/diagnostický nástroj za 0 tokenů a načte přes `read_file` jen konkrétní úsek řádků (40–100 řádků), kde leží problém.
2. **Konec slepých optimalizací na GPU:** Vektorní přepis jedné sekce smyčky nepomůže, pokud hlavní čas tiku mizí v Python dispatch overheadu, opakovaných GPU alokacích nebo skrytých `cudaStreamSynchronize` stallech v jiné části `_step_internal()`. Potřebujeme přesné rozlišení mezi **časem odeslání z CPU (`cpu_dispatch_ms`)** a **čistým časem běhu kernelů na GPU (`gpu_kernel_ms`)**.
3. **Bezpečná příprava na nadcházející modularizaci `engine_core.py`:** Před rozřezáním monolitického `RealisticBattleEngine` na samostatné subsystémy potřebujeme automaticky mapovat, které atributy `self.*` jednotlivé metody čtou a zapisují (`READS` / `WRITES`), a mít deterministický regresní zámek (`bench --compare`), který okamžitě odhalí jakoukoliv změnu chování nebo pokles TPS.
4. **Inspekce evoluce a `.pt` modelů bez spouštění GUI:** Umožnit rychlý náhled do uložených snapshotů (epochy, skóre frakcí, preference generálů, zdraví vah) i datových kontraktů tenzorů z příkazové řádky.

---

## 2. Architektonická pravidla a mantinely (Povinné)

* **Nulové nové závislosti:** Použít výhradně standardní knihovnu Pythonu (`ast`, `argparse`, `json`, `time`, `subprocess`, `pathlib`, `os`, `sys`) a již nainstalovaný `torch` (volitelně detekovat `ruff`, pokud je v `.venv`, jinak fallback na vlastní kontrolu).
* **Přísně Read-Only vůči zdrojovému kódu:** Žádný příkaz nesmí měnit `.py` soubory ani Git stav. Jediné povolené zápisy jsou `scripts/baseline_bench.json` (výhradně při explicitním `--save-baseline`) a úklid dočasných testovacích artefaktů v `health`.
* **Nulové znečištění horké smyčky:** Produkční kód (`src/engine_core.py` atd.) nesmí obsahovat žádné větve typu `if profiling:`. Měření probíhá neinvazivně za běhu v `perf_runner.py`.
* **Žádná zavádějící matematika u CUDA měření:** Nikdy nepočítat `overhead = cpu_wall - gpu_sum` (CPU a GPU běží asynchronně s překryvem). Vždy reportovat odděleně `cpu_dispatch_ms` a `gpu_kernel_ms`.

---

## 3. Cílová struktura souborů

```text
scripts/
├── aegis_tool.py          # Jednotný CLI vstupní bod (subcommands: xray, profile, bench, snaps, health)
└── tools/
    ├── xray.py            # Statická AST analýza kódu (složitost, sekční kotvy, hot-loop syncy, self.* R/W mapa, locks)
    ├── perf_runner.py     # Sdílený engine runner: GPU/CPU profiler, VRAM alloc audit, tensor contracts & bench guard
    └── snap_inspector.py  # Inspekce a porovnání uložených .pt snapshotů z telemetry/

.agent/skills/aegis-tooling/
└── SKILL.md               # Agentní manuál: povinný diagnostický protokol před čtením souborů
```

*(Poznámka: Pokud v repozitáři existují starší jednorázové skripty `ast_rentgen.py`, `spagety_metr.py` nebo `scan.py`, nechat je netknuté nebo je označit jako zastaralé, dokud nebude nový `aegis_tool.py` plně ověřen.)*

---

## 4. Detailní specifikace modulů a příkazů

### 4.1 `aegis_tool.py xray` (`scripts/tools/xray.py`)
Statický AST rentgen projektu (`src/`, `launcher.py`, `run_headless.py`, `tests/`), navržený tak, aby agentovi dal přesné souřadnice v kódu:

1. **Top-N nejsložitějších funkcí + vnitřní sekční kotvy:**
   * Pro každou funkci/metodu spočítat `start_line:end_line`, `LOC`, cyklomatickou složitost (`CC`) a maximální hloubku zanoření (`depth`).
   * **Klíčová funkce pro velké metody (> 100 řádků):** Extrahovat orientační kotvy uvnitř těla funkce (komentáře sekcí `# --- ... ---` nebo hlavní `if`/`for` bloky s jejich čísly řádků), aby agent nemusel číst 400 řádků najednou.
2. **Detekce GPU anti-patternů v hot-path (`src/engine_core.py`, `src/engine/*`):**
   * Vyhledat volání `.item()`, `.cpu()`, `.numpy()`, `.tolist()`, `.any()`, `.all()`, `torch.cuda.synchronize()` a dynamické indexování `torch.where(mask)` (které brání budoucímu `torch.compile`).
   * Označit, zda leží **uvnitř `for`/`while` smyčky** (`HOT_LOOP_SYNC`) nebo mimo ni.
3. **Mapa čtení a zápisů atributů (`--deps <třída_nebo_soubor>`):**
   * Pro každou metodu v `RealisticBattleEngine` (nebo zadané třídě) vypsat množinu čtených (`READS: self.pos, self.hp...`) a zapisovaných (`WRITES: self.fitness, self.nodes_gpu...`) atributů. Nezbytné pro bezpečné rozdělení `engine_core.py` na subsystémy.
4. **Kódová hygiena a zámky:**
   * Mapa všech `with .*lock:` bloků (soubor + řádek + název zámku) včetně detekce vnořených zámků.
   * Detekce tichých `except Exception: pass/continue` a nevolaných metod na subsystémech (`0` referencí v repozitáři).
   * Kontrola nepoužitých importů (`F401`) – primárně delegovat na `ruff check --select F401 --output-format json` (pokud je v `.venv`), jinak použít vlastní AST kontrolu s ošetřením `__future__`, `TYPE_CHECKING` a `__all__`.
5. **CLI přepínače:**
   * `--compact` (výstup do 55 řádků pro LLM),
   * `--deps <ClassName>` (výpis `READS`/`WRITES` mapy atributů),
   * `--json` (kompletní strojový výstup),
   * `--strict` (vrátí exit code `1` při nálezu `F401` nebo zakázaného GPU syncu uvnitř smyčky v hot-path).

---

### 4.2 `aegis_tool.py profile` (`scripts/tools/perf_runner.py`)
Neinvazivní sekční profiler simulačního tiku na GPU (`cuda:0`):

1. **Konfigurace běhu:**
   * Inicializovat `RealisticBattleEngine(2400, 1350, device="cuda:0")`, `visual_mode=False`, pevný `seed=42`, `N=800` jednotek.
   * Provést `30` warm-up tiků (zahřátí CUDA kontextu a caching alokátoru) a následně měřit `200` tiků.
2. **Správná metodika měření CPU vs. GPU:**
   * Za běhu obalit klíčové metody subsystémů (`spatial.resolve_obstacle_collisions`, `spatial.check_line_of_sight`, `brains.forward`, `combat.process_abilities`, `ballistics.resolve_ballistic_hits`, `objectives.update_gates`, `objectives.check_battle_status`, `_atomic_commit_pipeline`).
   * Uvnitř wrapperu zaznamenat:
     * `start_evt.record()` → `t0 = time.perf_counter()` → `res = orig(*args, **kwargs)` → `t1 = time.perf_counter()` → `end_evt.record()`.
     * **ZÁKAZ volání `torch.cuda.synchronize()` uvnitř wrapperů!** Synchronizace proběhne pouze jednou na konci celého tiku.
   * Reportovat pro každou sekci:
     * **`cpu_dispatch_ms`** (`t1 - t0`): pokud je vysoké (> 0,5 ms), daná metoda na CPU tajně čeká na GPU (`cudaStreamSynchronize`) nebo dělá těžkou Python smyčku.
     * **`gpu_kernel_ms`** (`start_evt.elapsed_time(end_evt)`): skutečný výpočetní čas kernelů daného subsystému na RTX 5070 Ti.
3. **VRAM & Alokační audit (`torch.cuda.memory_stats()`):**
   * Změřit přírůstek alokací přes `100` tiků po warm-upu a vypsat průměrný počet volání alokátoru na tik (`alloc_calls_per_tick`) a špičkovou paměť (`peak_vram_mb`). Vysoký počet alokací odhalí místa, kde se v každém tiku zbytečně vytvářejí nové dočasné tenzory.

---

### 4.3 `aegis_tool.py bench` (`scripts/tools/perf_runner.py`)
Deterministický hlídač výkonu, herní logiky a datových kontraktů:

1. **Konfigurace běhu:**
   * Čistý běh **bez profilovacích wrapperů** (nulová režie měření): `seed=42`, `N=800`, `ticks=500`.
   * Vypočítat `total_sec`, `ms_per_tick` a `tps`.
2. **Kontrola datových kontraktů a numerické stability (Contract Check):**
   * Po doběhnutí `500` tiků (a 1 vynuceném běhu `finish_operation_and_evolve()`) ověřit:
     * správné tvary (`shape`), datové typy (`dtype`) a umístění na `cuda:0` u klíčových tenzorů (`pos`, `vel`, `hp`, `teams`, `classes`, `fitness`, `nodes_gpu`, `W1`, `W2`, `W3`),
     * **nulový výskyt `NaN` nebo `Inf`** ve všech stavových tenzorech a vahách sítí.
3. **Dvoustupňové ověření shody logiky (odolné vůči GPU `float32` driftu):**
   * Paralelní atomické operace (`index_add_`, redukce) na CUDA mohou způsobit odchylku v řádu `1e-5`, která u prahových podmínek (`hp <= 0.0`, `trigger > 0.58`) výjimečně posune diskrétní počet o `±1`.
   * Proto kontrola rozlišuje tři stavy:
     * **`EXACT`**: 100% shoda diskrétních hodnot (`tick_counter`, `alive_count`, `kills_per_faction`, `node_owners`) a spojitých hodnot v rámci `rtol=1e-4, atol=1e-4`.
     * **`WITHIN_TOL`**: diskrétní počty v úzké toleranci (`±2` jednotky na 500 ticích) a spojité statistiky (`node_control_points`, `sum/mean/std` pro `pos`, `hp`, `fitness`) v rámci `rtol=1e-3, atol=1e-2`.
     * **`REGRESSION`**: překročení tolerance nebo chyba kontraktu (`NaN`/špatný tvar) → exit code `1`.
4. **CLI přepínače:**
   * `--save-baseline`: uloží referenční JSON do `scripts/baseline_bench.json`.
   * `--compare`: porovná aktuální běh proti baseline a vypíše jednořádkový verdikt.

---

### 4.4 `aegis_tool.py snaps` (`scripts/tools/snap_inspector.py`)
Inspektor uložených `.pt` modelů v adresáři `telemetry/` (pro vývojářku i agenta):

* `aegis_tool.py snaps list`: přehledná tabulka uložených snapshotů (název souboru, velikost v MB, epocha, scénář, počet výher frakcí, datum).
* `aegis_tool.py snaps inspect <cesta_k_pt>`: načte `.pt` na CPU (`map_location="cpu"`, `weights_only=False`) a vypíše:
  * XP, levely a váhy preferencí tříd/technologií jednotlivých `FactionGeneralBrain`,
  * zdravotní statistiku genetických bank `W1..W3` (min/max/std vah, procento téměř nulových/mrtvých vah, detekce `NaN`/`Inf`),
  * ověření kompatibility klíčů (`_full` vs. legacy formát).
* `aegis_tool.py snaps diff <snap_a.pt> <snap_b.pt>`: porovná posun XP generálů, ROI tříd a průměrné/nejlepší fitness mezi dvěma epochami.

---

### 4.5 `aegis_tool.py health` (`scripts/aegis_tool.py`)
Jednopříkazový pre-commit gate před každým commitem:

1. `python -m py_compile` na všechny `.py` soubory v repozitáři.
2. `aegis_tool.py xray --strict` (`0` nepoužitých importů, `0` GPU synců uvnitř smyček v hot-path).
3. Spuštění `tests/test_suite.py` na `cuda:0` + bezpečný úklid dočasných testovacích logů vygenerovaných během testu.
4. Krátký `git status --short`.

---

## 5. Vzorové `--compact` výstupy (optimalizováno pro LLM kontext)

### `python scripts/aegis_tool.py xray --compact`
```text
[AEGIS XRAY] files=14 loc=6420 f401=0 hot_loop_sync=0 silent_except=2
TOP_COMPLEXITY (lines / LOC / CC / depth):
  1. src/engine_core.py:508-920 _step_internal (loc=412, cc=48, depth=5)
     anchors: 515:battle_check | 610:sensors | 745:movement | 860:abilities
  2. src/tactical_hud.py:310-595 render (loc=285, cc=34, depth=4)
     anchors: 325:map_bg | 410:units | 520:inspector_panel
DEAD_METHODS:
  src/engine/objectives_system.py:88 ObjectivesSystem.update_control_nodes (0 refs)
LOCKS:
  src/engine_core.py:505(_engine_lock) | src/engine/async_runner.py:83,104(_engine_lock)
```

### `python scripts/aegis_tool.py profile --compact`
```text
[AEGIS PROFILE] N=800 warmup=30 ticks=200 device=cuda:0 (RTX 5070 Ti)
AVG_TICK: wall=16.84ms | VRAM_DELTA: peak=+1.2MB | alloc_calls_per_tick=34
SUBSYSTEMS (cpu_dispatch_ms / gpu_kernel_ms):
  1. uninstrumented_step_body (sensors/cdist/move) : 11.40ms / --
  2. brains.forward                                :  0.32ms / 0.65ms
  3. spatial.resolve_obstacle_collisions           :  0.28ms / 0.54ms
  4. ballistics.resolve_ballistic_hits             :  0.24ms / 0.48ms
  5. _atomic_commit_pipeline                       :  0.19ms / 0.18ms
  6. objectives.update_gates                       :  0.11ms / 0.12ms
```
*(Poznámka: Vysoké `cpu_dispatch_ms` u `uninstrumented_step_body` okamžitě ukáže, že další optimalizace má mířit přímo na výpočet senzorů / `cdist` / pohyb v těle `_step_internal()`, nikoli do `brains` nebo `ballistics`.)*

---

## 6. Agentní vrstva (`.agent/skills/aegis-tooling/SKILL.md`)

Vytvořit workspace skill s tímto závazným postupem:
1. **Diagnostika před čtením:** Při každém požadavku na audit, optimalizaci, refaktoring nebo ladění chyby nejprve spusť `python scripts/aegis_tool.py xray --compact` (případně `profile --compact` nebo `xray --deps RealisticBattleEngine`).
2. **Cílené čtení řádků:** Podle čísel řádků a sekčních kotev z výstupu načti přes `read_file` pouze konkrétní funkci nebo blok (max. ~100 řádků), nikdy celý 1 000+ řádkový soubor.
3. **Povinná regresní verifikace:** Po každé úpravě kódu spusť `python scripts/aegis_tool.py health`. Při zásahu do simulační logiky nebo výkonu nejprve ulož baseline (`bench --save-baseline`) před úpravou a po úpravě ověř přes `bench --compare`.

---

## 7. Akceptační kritéria a ověření implementace

- [ ] Všechny 4 moduly (`scripts/aegis_tool.py`, `scripts/tools/xray.py`, `scripts/tools/perf_runner.py`, `scripts/tools/snap_inspector.py`) + `.agent/skills/aegis-tooling/SKILL.md` existují a mají nulové externí závislosti mimo `.venv`.
- [ ] `python scripts/aegis_tool.py xray --compact` i `--deps RealisticBattleEngine` proběhnou pod **1,5 s** a přesně zobrazí sekční kotvy i `READS`/`WRITES` atributy.
- [ ] `python scripts/aegis_tool.py profile --compact` korektně měří `cpu_dispatch_ms`, `gpu_kernel_ms` (bez synchronizace uvnitř wrapperů) a `alloc_calls_per_tick` na `cuda:0`.
- [ ] `python scripts/aegis_tool.py bench --save-baseline` a následný `bench --compare` na nezměněném kódu vrátí stav `EXACT` a `0` `NaN`/`Inf` v tenzorech.
- [ ] `python scripts/aegis_tool.py snaps list` a `snaps inspect` fungují nad reálnými `.pt` soubory v `telemetry/` (včetně zpětné kompatibility).
- [ ] `python scripts/aegis_tool.py health` projde se status kódem `0` a nezanechá po sobě žádné dočasné testovací soubory.