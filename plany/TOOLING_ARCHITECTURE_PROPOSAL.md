# Návrh architektury: Podpůrné diagnostické nástroje & Agentní Tooling (Aegis Tooling Kit)

> **Účel dokumentu:** Technický návrh pro křížový audit a validaci před implementací.
> **Cílový projekt:** Cyber-Tactics 2.0 (PyTorch/CUDA neuroevoluční simulátor, Pygame HUD, Python 3.12, Windows, RTX 5070 Ti).

---

## 1. Motivace a hlavní cíle

Při vývoji a postupném refaktoringu („od chaosu ku řádu“) narážíme na dva praktické problémy:
1. **Spotřeba kvót AI agentů:** Při každém auditu nebo hledání chyby agent zbytečně čte celé 1 000+ řádkové soubory (`engine_core.py`, `launcher.py`, `tactical_hud.py`) do kontextového okna.
2. **Slepé optimalizace:** Jak ukázala Fáze 2 (vektorizace uzlů a killů zrychlila tik z `17,035 ms` na `16,840 ms`, přestože se logika zjednodušila), bez přesného měření po sekcích nevíme, kde mizí zbývajících ~16 ms na GPU.

**Cíl:** Vytvořit malou, soběstačnou sadu lokálních Python nástrojů (nulové nové závislosti mimo stdlib a existující `.venv`) s jednotným CLI rozhraním, které:
* běží lokálně za **0 tokenů**,
* produkují **deterministická, tvrdá data** (AST metriky, CUDA časy, regresní hashe),
* mají režim `--compact` / `--json` navržený přímo pro AI agenty (úspora 80–90 % kontextu).

---

## 2. Dvouvrstvá architektura toolingu

```text
scripts/
├── aegis_tool.py          # Jednotný CLI vstupní bod (subcommands: xray, profile, bench, health)
└── tools/
    ├── xray.py            # Statická AST analýza kódu (nástupce ast_rentgen + spagety_metr + scan)
    ├── profiler_gpu.py    # Sekční GPU/CPU profiler simulačního tiku (torch.cuda.Event)
    └── bench_guard.py     # Deterministický benchmark + porovnání oproti baseline JSON

.agent/skills/aegis-tooling/
└── SKILL.md               # Agentní manuál: "Nečti celé soubory, spusť aegis_tool.py!"
```

*(Staré jednorázové skripty `ast_rentgen.py`, `spagety_metr.py`, `scan.py` lze po ověření nového toolu buď archivovat, nebo nechat jako tenké obálky.)*

---

## 3. Detailní specifikace jednotlivých modulů

### 3.1 `aegis_tool.py xray` — Statický AST rentgen
Nahrazuje ruční procházení souborů a sjednocuje statické kontroly do jednoho běhu (< 1 s):
* **Metriky složitosti:** LOC, počet tříd/funkcí, cyklomatická složitost (CC), maximální hloubka zanoření (vypisuje jen top-N problémových funkcí).
* **GPU anti-patterny:** vyhledá všechny `.item()`, `.cpu()`, `.numpy()`, `.tolist()` a `torch.cuda.synchronize()` a **označí ty, které leží uvnitř `for`/`while` smyček** v hot-path souborech (`src/engine_core.py`, `src/engine/*`).
* **Kvalita & hygiena:** nepoužité importy (`F401`), tiché `except Exception: pass/continue`, nevolané metody v subsystémech (např. opuštěný `ObjectivesSystem.update_control_nodes`).
* **Mapa závislostí a zámků:** seznam interních importů (včetně detekce nekonzistence `src.X` vs `X`) a přehled všech `with .*lock:` bloků.
* **Režimy:** `--compact` (stručná tabulka do 60 řádků pro LLM), `--json` (strojově čitelné), `--strict` (vrátí exit code `1` při novém `F401` nebo `.item()` v hot-loopu).

### 3.2 `aegis_tool.py profile` — Sekční GPU/CPU profiler tiku
Odpovídá na otázku: **„Kde přesně mizí 16,8 ms v jednom tiku při 800 jednotkách?“**
* Spustí `RealisticBattleEngine(2400, 1350, cuda:0)` v headless režimu (`visual_mode=False`, pevný seed, warm-up 30 tiků, měření 200 tiků).
* Měří **bez úpravy produkčního kódu** (nebo přes lehký volitelný hook / monkey-patch subsystémů) rozpad času na:
  1. `spatial.resolve_obstacle_collisions` & `check_line_of_sight`
  2. `brains.forward` (MLP inference)
  3. `combat.process_abilities`
  4. `ballistics.resolve_ballistic_hits`
  5. `objectives.update_gates` & `check_battle_status`
  6. `_atomic_commit_pipeline`
  7. Zbytek `_step_internal` (senzory, pohyb, `cdist`, uzly)
* **Technický detail:** Pro přesnost na GPU použije páry `torch.cuda.Event(enable_timing=True)` + CPU `time.perf_counter()`, takže odliší čistý čas GPU kernelu od CPU režie/synchronizačních stallů.

### 3.3 `aegis_tool.py bench` — Hlídač determinismu a výkonu (`bench_guard`)
Zajišťuje, že žádná budoucí optimalizace nerozbije herní logiku ani nezpomalí simulaci:
* Spustí fixní scénář (např. `seed=42`, `N=800`, `ticks=500` nebo `1000`).
* Vypočítá: `total_sec`, `ms_per_tick`, `tps`, `kills_per_faction`, `node_control_points`, a deterministický kontrolní součet (hash zaokrouhlených pozic/HP/fitness).
* **`--save-baseline`:** uloží referenční výsledek do `scripts/baseline_bench.json` (ignorováno nebo verzováno dle volby).
* **`--compare`:** porovná aktuální běh proti baseline a vypíše jednořádkový verdikt:
  * `TPS: 59.4 -> 142.0 (+139.1%) | LOGIC MATCH: 100% (EXACT)`

### 3.4 `aegis_tool.py health` — Jednopříkazový pre-commit gate
Spustí vše důležité před každým commitem v jednom volání:
1. `python -m py_compile` na všechny aktivní `.py` soubory.
2. `aegis_tool.py xray --strict` (0 nepoužitých importů, 0 `.item()` ve smyčkách hot-loopu).
3. `tests/test_suite.py` (všech 9+ testů na `cuda:0`) + automatický úklid testovacích logů z `telemetry/`.
4. Krátký `git status --short`.

---

## 4. Agentní vrstva (Workspace Skill pro úsporu kvót)

Soubor `.agent/skills/aegis-tooling/SKILL.md` dá každému budoucímu AI agentovi jasný protokol:
1. **Vždy začni nástrojem, ne čtením souborů:** Při požadavku na audit, optimalizaci nebo debug nejdřív spusť `python scripts/aegis_tool.py xray --compact` nebo `profile --compact`.
2. **Čti jen cílené řádky:** Na základě čísel řádků z `xray`/`profile` načti přes `read_file` jen konkrétní funkci (např. 40–80 řádků), nikdy celý 1 700řádkový soubor.
3. **Ověřuj přes `health` a `bench --compare`:** Žádná výkonová změna se neprohlašuje za „rychlejší“ bez čísla z `bench --compare`.

---

## 5. Designové mantinely

* **Read-only by default:** Žádný z nástrojů nikdy nemění zdrojové soubory ani Git (jediný zápis je volitelný `baseline_bench.json` při `--save-baseline`).
* **Nulové nové závislosti:** Pouze Python stdlib (`ast`, `argparse`, `json`, `time`, `hashlib`, `subprocess`) + již nainstalovaný `torch`.
* **Žádné znečištění produkčního kódu:** Profiler nesmí přidávat žádné `if profiling:` větve do horké smyčky `engine_core.py`, když se nepoužívá.

---

## 6. Otázky pro křížový audit (pro Gemini na webu)

1. **Není návrh zbytečně překombinovaný (overengineering)?** Kterou část bys sloučil nebo zjednodušil, aby měla nejvyšší poměr přínos/složitost?
2. **Měření GPU v `profiler_gpu.py`:** Je lepší obalit subsystémy (`spatial`, `ballistics`, `combat`, `brains`) tenkým monkey-patch wrapperem s `torch.cuda.Event`, nebo použít vestavěný `torch.profiler.profile(activities=[CPU, CUDA])` s agregací tabulky? Jaké jsou výhody a nevýhody obou přístupů pro Windows + PyTorch 2.12 nightly?
3. **Determinismus na CUDA (`bench_guard.py`):** Stačí `torch.manual_seed(42)` + `torch.cuda.manual_seed_all(42)`, nebo mohou nedeterministické atomické operace (`index_add_` na GPU) způsobit drobné rozdíly v řádu `1e-5` při porovnání hashe? (Návrh: porovnávat s tolerancí `rtol/atol` místo tvrdého SHA256 hashe floatů.)
4. **Formát výstupu pro LLM:** Co přesně by měl obsahovat `--compact` výstup z `xray` a `profile`, aby agent dostal maximum informace v co nejmenším počtu tokenů?
