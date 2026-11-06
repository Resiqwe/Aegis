# test

Experimentální projekt: softwarové 4bitové CPU postavené z jednotlivých neuronů (Python, NumPy). Učení a experimenty, ne produkční kód.

## Struktura
| Složka | Obsah |
|---|---|
| `neuron/` | `neuron_node.py` (základní buňka, dumpy vah, audit, hash řetěz), `chain_runner.py` (řetězení buněk, L1 Transit Buffer) |
| `cpu/` | stavba CPU: `cpu_step1_xor` až `cpu_step4_core`, `cpu_telemetry_probe`, `cpu_hardware_clock_core` (TuringNeuronCPU, hodiny, JZ/JNZ) |
| `demos/` | programy nad CPU: Fibonacci, Gaussův součet, 8bitové ADC, telemetrie |
| `tests/` | `test_torture_suite.py` (buňka, viz `METODIKA_TESTU.md`), `cpu_torture_suite.py` (CPU) |
| `archiv/` | starý kód (`_historie/`, `turtle_*.py`), jen pro referenci |
| `ukoly/` | zadání agentům (`SABLONA.md`) |
| `dumps/`, `logs/` | generované výstupy, ignorované gitem |

## Spouštění
Z kořene projektu `test/`, vždy jako modul (balíčky):
```
.\.venv\Scripts\Activate.ps1
python -m tests.test_torture_suite
python -m tests.cpu_torture_suite
python -m demos.cpu_fibonacci_long_run
```
Stav testů po přesunu struktury zatím neověřen.