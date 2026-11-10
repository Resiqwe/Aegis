"""
tune_core.py - Vytunění základního kamínku: Hledání minimálních architektur
pro všechny základní logické a aritmetické operandy čipu:
  1. NOT  (1 vstup -> 1 výstup)
  2. AND  (2 vstupy -> 1 výstup)
  3. OR   (2 vstupy -> 1 výstup)
  4. NAND (2 vstupy -> 1 výstup)
  5. NOR  (2 vstupy -> 1 výstup)
  6. XOR  (2 vstupy -> 1 výstup)
  7. FULL_ADDER (3 vstupy -> 2 výstupy) - Sčítací jednotka

Cíl: Najít nejmenší možný počet neuronů a vah, otestovat 100% spolehlivost
a uložit výsledné "cucky vah" do bleskového zmrazeného ROM slovníku.
"""

import sys
import time
import itertools
import numpy as np
from brick import Brick

# Definice pravdivostních tabulek pro všechny operandy
OPERANDS = {
    "NOT": {
        "n_in": 1, "n_out": 1,
        "fn": lambda a: (1 - a,),
        "test_hidden": [[]] # NOT zvládne 1 neuron bez skryté vrstvy
    },
    "AND": {
        "n_in": 2, "n_out": 1,
        "fn": lambda a, b: (a & b,),
        "test_hidden": [[]] # AND zvládne 1 neuron
    },
    "OR": {
        "n_in": 2, "n_out": 1,
        "fn": lambda a, b: (a | b,),
        "test_hidden": [[]] # OR zvládne 1 neuron
    },
    "NAND": {
        "n_in": 2, "n_out": 1,
        "fn": lambda a, b: (1 - (a & b),),
        "test_hidden": [[]] # NAND zvládne 1 neuron
    },
    "NOR": {
        "n_in": 2, "n_out": 1,
        "fn": lambda a, b: (1 - (a | b),),
        "test_hidden": [[]] # NOR zvládne 1 neuron
    },
    "XOR": {
        "n_in": 2, "n_out": 1,
        "fn": lambda a, b: (a ^ b,),
        "test_hidden": [[2], [3], [4]] # Hledáme minimum: stačí 2 skryté neurony?
    },
    "FULL_ADDER": {
        "n_in": 3, "n_out": 2,
        "fn": lambda a, b, cin: ((a + b + cin) & 1, (a + b + cin) >> 1),
        "test_hidden": [[3], [4], [5], [6]] # Hledáme absolutní minimum pro sčítačku
    }
}

class FastFrozenBrick:
    """
    Extrémně rychlý, zmrazený blok. Žádný trénink, žádné objekty.
    Jen čisté maticové násobení přes natrénované váhy.
    """
    def __init__(self, name: str, params: list):
        self.name = name
        self.params = params
        self.n_in = params[0][0].shape[0]

    def __call__(self, *bits) -> tuple:
        x = np.array([bits], dtype=np.float64)
        for i, (W, b) in enumerate(self.params):
            z = x @ W + b
            x = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))) if i == len(self.params) - 1 else np.tanh(z)
        return tuple(int(v >= 0.5) for v in x[0])


def tune_all():
    print("=" * 80)
    print(" TUNING ZÁKLADNÍCH KAMÍNKŮ: HLEDÁNÍ MINIMÁLNÍCH ARCHITEKTUR")
    print("=" * 80)
    
    frozen_chips = {}
    
    for name, spec in OPERANDS.items():
        n_in = spec["n_in"]
        n_out = spec["n_out"]
        fn = spec["fn"]
        
        found = False
        for hidden in spec["test_hidden"]:
            try:
                # Zkusíme natrénovat s touto mini-architekturou
                brick = Brick.learn(
                    name=name,
                    n_in=n_in,
                    n_out=n_out,
                    fn=fn,
                    hidden=hidden,
                    max_epochs=4000,
                    lr=0.08,
                    attempts=15
                )
                st = brick.stats()
                frozen = FastFrozenBrick(name, brick.params)
                
                # Vyčerpávající verifikace
                for bits in itertools.product((0, 1), repeat=n_in):
                    assert frozen(*bits) == fn(*bits), f"Chyba na vstupu {bits}!"
                
                frozen_chips[name] = frozen
                h_str = str(hidden) if hidden else "0 (Lineární)"
                print(f"  [✓] {name:<11} | Skryté: {h_str:<12} | Neurony: {st['neurons']:<2} | Vah celkem: {st['params']:<3} | 100% OK")
                found = True
                break
            except RuntimeError:
                continue
                
        if not found:
            print(f"  [!] {name:<11} | Nepodařilo se najít stabilní váhy v testovaném rozsahu!")

    # Test bleskové propustnosti (kolik milionů operací za sekundu zvládne proletět)
    print("\n" + "=" * 80)
    print(" TEST RYCHLOSTI PRŮLETU VAHAMI (10 000 operací za sebou):")
    print("=" * 80)
    
    fa = frozen_chips["FULL_ADDER"]
    t0 = time.perf_counter()
    for _ in range(10_000):
        _ = fa(1, 0, 1)
    dt = time.perf_counter() - t0
    ops_per_sec = 10_000 / dt
    
    print(f"  Plná sčítačka (FULL_ADDER): {ops_per_sec:,.0f} operací/sekundu")
    print(f"  Čas jednoho průletu:        {dt * 100:,.2f} mikrosekund (čistý Python bez GPU/C)")
    print("=" * 80)
    
    return frozen_chips

if __name__ == "__main__":
    tune_all()
