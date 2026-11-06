# test

Experimentální projekt: z neuronů se staví logika, bloky a časem čip. Učení a experimenty, ne produkční kód. Jen NumPy.

## Aktuální linie (nová, čistá)
| Soubor | Co to je |
|---|---|
| `perceptron.py` | mikro-knihovnička: `Perceptron`, hradla (`Gates`), `add`, `sub`, `SRLatch`. Samokontrola: `python perceptron.py` |
| `brick.py` | "lego kostička": malá síť, která se naučí celý blok z pravdivostní tabulky a je ověřena vyčerpávajícím testem. Samokontrola: `python brick.py` |

```python
from perceptron import Perceptron, Gates, add
from brick import Brick
```

## Starší linie (dědictví, jen pro referenci)
| Složka | Obsah |
|---|---|
| `neuron/` | původní `neuron_node.py` (dumpy vah, hash řetěz, audit) a `chain_runner.py` |
| `cpu/` | pokusy o CPU (step1-4, telemetrie, hodiny). Velká část je obyčejný Python (SUB, RAM, skoky) |
| `demos/` | programy nad starým CPU. `cpu_fibonacci_long_run.py` je rozbitý, viz `archiv/` |
| `tests/` | torture sady pro starou linii |
| `archiv/` | úplně staré soubory (`_historie/`, `turtle_*.py`) |
| `ukoly/` | zadání agentům (`SABLONA.md`) |
| `dumps/`, `logs/` | generované výstupy, ignorované gitem |

Starou linii spouštět z kořene jako moduly, např. `python -m tests.cpu_torture_suite`.
