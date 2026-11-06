"""
brick.py - první "lego kostička": malá neuronová síť, která se naučí CELÝ blok najednou.

Kostička má jednotné rozhraní, takže z ní půjde skládat vyšší vrstvy:
    bity dovnitř (n_in) -> bity ven (n_out)

Použití:
    from brick import Brick

    xor = Brick.learn("XOR", n_in=2, n_out=1, fn=lambda a, b: (a ^ b,), hidden=[4])
    xor(1, 0)            # -> (1,)
    xor.stats()          # neurony, parametry, epochy, čas, ověřeno

    fa = Brick.learn("FULL_ADDER", 3, 2, lambda a, b, c: ((a + b + c) & 1, (a + b + c) >> 1), hidden=[6])
    fa(1, 1, 1)          # -> (1, 1)   (suma, carry)

Kostička se učí z celé pravdivostní tabulky (všech 2**n_in vstupů) a po naučení se VYČERPÁVAJÍCÍM
testem ověří. Bez úspěšného ověření se nevrátí (RuntimeError). Po naučení je zmrazená (jen čtení).

Samokontrola:  python brick.py   (exit 0 = OK, exit 1 = chyba)
Jen NumPy, žádné soubory na disku, žádný globální stav.
"""

import itertools
import sys
import time

import numpy as np


class Brick:
    def __init__(self, name: str, n_in: int, n_out: int, hidden: list[int], seed: int = 0):
        self.name = name
        self.n_in, self.n_out, self.hidden = n_in, n_out, list(hidden)
        sizes = [n_in, *self.hidden, n_out]
        rng = np.random.default_rng(seed)
        self.params = [
            [rng.standard_normal((a, b)) * np.sqrt(1.0 / a), np.zeros(b)]
            for a, b in zip(sizes[:-1], sizes[1:])
        ]
        self.info = {"epochs": 0, "seconds": 0.0, "verified": False, "attempts": 0}

    # --- výpočet (skryté vrstvy tanh, výstup sigmoida) ---
    def _forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(self.params):
            z = acts[-1] @ W + b
            acts.append(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))) if i == len(self.params) - 1 else np.tanh(z))
        return acts

    def __call__(self, *bits) -> tuple:
        if len(bits) != self.n_in:
            raise ValueError(f"{self.name}: očekáváno {self.n_in} vstupů, dáno {len(bits)}")
        out = self._forward(np.array([bits], dtype=np.float64))[-1][0]
        return tuple(int(v >= 0.5) for v in out)

    # --- učení: backpropagation + Adam, plná dávka ---
    def _fit(self, X, Y, max_epochs: int, lr: float) -> int:
        m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in self.params]
        v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in self.params]
        n = len(X)
        for epoch in range(1, max_epochs + 1):
            acts = self._forward(X)
            delta = (acts[-1] - Y) / n                     # gradient BCE + sigmoida
            for i in range(len(self.params) - 1, -1, -1):
                gW, gb = acts[i].T @ delta, delta.sum(axis=0)
                if i > 0:
                    delta = (delta @ self.params[i][0].T) * (1.0 - acts[i] ** 2)   # derivace tanh
                for k, g in enumerate((gW, gb)):
                    m[i][k] = 0.9 * m[i][k] + 0.1 * g
                    v[i][k] = 0.999 * v[i][k] + 0.001 * g * g
                    mh = m[i][k] / (1 - 0.9 ** epoch)
                    vh = v[i][k] / (1 - 0.999 ** epoch)
                    self.params[i][k] -= lr * mh / (np.sqrt(vh) + 1e-8)
            if epoch % 50 == 0 and np.array_equal((self._forward(X)[-1] >= 0.5).astype(int), Y.astype(int)):
                return epoch
        return max_epochs

    def verify(self, table) -> bool:
        """Vyčerpávající kontrola: každý vstup musí dát přesně očekávaný výstup."""
        return all(self(*bits) == tuple(out) for bits, out in table)

    @classmethod
    def learn(cls, name, n_in, n_out, fn, hidden, max_epochs=5000, lr=0.05, attempts=20, seed=0):
        """Naučí kostičku z funkce fn(*bity) -> tuple bitů. Zkouší různé počáteční váhy (seed)."""
        table = [(bits, tuple(fn(*bits))) for bits in itertools.product((0, 1), repeat=n_in)]
        X = np.array([b for b, _ in table], dtype=np.float64)
        Y = np.array([o for _, o in table], dtype=np.float64)
        start = time.perf_counter()
        for attempt in range(1, attempts + 1):
            brick = cls(name, n_in, n_out, hidden, seed=seed + attempt - 1)
            epochs = brick._fit(X, Y, max_epochs, lr)
            if brick.verify(table):
                brick.info = {"epochs": epochs, "seconds": time.perf_counter() - start,
                              "verified": True, "attempts": attempt}
                return brick
        raise RuntimeError(f"{name}: po {attempts} pokusech se nenaučilo (hidden={hidden})")

    def stats(self) -> dict:
        neurons = sum(self.hidden) + self.n_out
        weights = sum(W.size + b.size for W, b in self.params)
        return {"name": self.name, "layout": [self.n_in, *self.hidden, self.n_out],
                "neurons": neurons, "params": weights, **self.info}


# ----------------------------------------------------------------- samokontrola
def _selfcheck() -> int:
    errors = []

    def check(ok, msg):
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            errors.append(msg)

    def row(b):
        s = b.stats()
        return f"{s['name']:<11} {s['layout']} neurony={s['neurons']} par.={s['params']} epochy={s['epochs']} pokusů={s['attempts']} {s['seconds']*1000:.0f} ms"

    specs = [
        ("AND", 2, 1, lambda a, b: (a & b,), []),
        ("NOR", 2, 1, lambda a, b: (1 - (a | b),), []),
        ("XOR", 2, 1, lambda a, b: (a ^ b,), [4]),
        ("FULL_ADDER", 3, 2, lambda a, b, c: ((a + b + c) & 1, (a + b + c) >> 1), [6]),
        ("ADD2", 4, 3, lambda a1, a0, b1, b0: tuple(((a1 * 2 + a0 + b1 * 2 + b0) >> i) & 1 for i in (2, 1, 0)), [10]),
    ]
    for name, n_in, n_out, fn, hidden in specs:
        try:
            b = Brick.learn(name, n_in, n_out, fn, hidden)
            check(b.info["verified"], row(b))
        except RuntimeError as e:
            check(False, str(e))

    try:
        Brick.learn("XOR_1_NEURON", 2, 1, lambda a, b: (a ^ b,), [], max_epochs=2000, attempts=3)
        check(False, "samotný neuron XOR nemá zvládnout")
    except RuntimeError:
        check(True, "XOR bez skryté vrstvy se nenaučí (očekávané)")

    try:
        Brick.learn("AND", 2, 1, lambda a, b: (a & b,), [])(1)
        check(False, "špatný počet vstupů má vyhodit výjimku")
    except ValueError:
        check(True, "špatný počet vstupů odmítnut")

    print("VÝSLEDEK:", "OK" if not errors else f"{len(errors)} CHYB")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(_selfcheck())
