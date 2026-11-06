"""
perceptron.py - mikro-knihovnička: perceptron a logika postavená z natrénovaných perceptronů.

Použití:
    from perceptron import Perceptron, Gates, SRLatch, add, sub

    p = Perceptron(2, seed=1)
    p.fit(X, y)
    p.predict(X)

    g = Gates()                  # natrénuje AND, OR, NAND, NOR (jednou)
    g.run("AND", 1, 1)           # -> 1
    add(g, 9, 5)                 # -> (14, 0)  (výsledek, carry)

Samokontrola:  python perceptron.py   (exit 0 = vše sedí, exit 1 = chyba)

Jen NumPy, žádný globální stav, žádné soubory na disku.
"""

import sys
import numpy as np


# ------------------------------------------------------------------ perceptron
class Perceptron:
    """Jeden neuron: y = sigmoid(X @ W + b), učení gradientním sestupem (binary cross-entropy)."""

    def __init__(self, n_inputs: int, lr: float = 1.5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal(n_inputs)
        self.b = 0.0
        self.lr = lr

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def predict_proba(self, X):
        return self._sigmoid(np.asarray(X, dtype=np.float64) @ self.W + self.b)

    def predict(self, X):
        """Tvrdé rozhodnutí 0/1 (práh 0.5)."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def fit(self, X, y, max_epochs: int = 20000, target_loss: float = 1e-3) -> dict:
        """Trénuje, dokud MSE nespadne pod target_loss nebo nedoběhnou epochy."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(X)
        loss = float(np.mean((self.predict_proba(X) - y) ** 2))
        epoch = 0
        while loss > target_loss and epoch < max_epochs:
            error = self.predict_proba(X) - y          # gradient BCE vůči z
            self.W -= self.lr * (X.T @ error) / n
            self.b -= self.lr * float(error.sum()) / n
            epoch += 1
            loss = float(np.mean((self.predict_proba(X) - y) ** 2))
        return {"epochs": epoch, "loss": loss, "converged": loss <= target_loss}


# ---------------------------------------------------------------------- hradla
_X2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
TRUTH = {
    "AND":  [0, 0, 0, 1],
    "OR":   [0, 1, 1, 1],
    "NAND": [1, 1, 1, 0],
    "NOR":  [1, 0, 0, 0],
}


class Gates:
    """Sada hradel; každé je jeden natrénovaný perceptron. XOR vzniká složením (jeden neuron XOR neumí)."""

    def __init__(self, seed: int = 0):
        self.neurons = {}
        for i, (name, y) in enumerate(TRUTH.items()):
            p = Perceptron(2, seed=seed + i)
            info = p.fit(_X2, y)
            if not info["converged"]:
                raise RuntimeError(f"Hradlo {name} se nenaučilo: {info}")
            self.neurons[name] = p

    def run(self, name: str, a: int, b: int) -> int:
        return int(self.neurons[name].predict([[a, b]])[0])

    def not_(self, a: int) -> int:
        return self.run("NAND", a, a)

    def xor(self, a: int, b: int) -> int:
        return self.run("AND", self.run("NAND", a, b), self.run("OR", a, b))


# ------------------------------------------------------------------- aritmetika
def full_adder(g: Gates, a: int, b: int, cin: int = 0) -> tuple[int, int]:
    s1 = g.xor(a, b)
    s = g.xor(s1, cin)
    cout = g.run("OR", g.run("AND", a, b), g.run("AND", s1, cin))
    return s, cout


def add(g: Gates, a: int, b: int, bits: int = 4, cin: int = 0) -> tuple[int, int]:
    """Sčítání přes řetěz plných sčítaček. Vrací (výsledek, carry)."""
    result, carry = 0, cin
    for i in range(bits):
        s, carry = full_adder(g, (a >> i) & 1, (b >> i) & 1, carry)
        result |= s << i
    return result, carry


def sub(g: Gates, a: int, b: int, bits: int = 4) -> tuple[int, int]:
    """Odčítání a - b jako a + ~b + 1. Vrací (výsledek, borrow); borrow=1 když a < b."""
    nb = 0
    for i in range(bits):
        nb |= g.not_((b >> i) & 1) << i
    result, carry = add(g, a, nb, bits, cin=1)
    return result, 1 - carry


# -------------------------------------------------------------------- paměť (1 bit)
class SRLatch:
    """SR klopný obvod ze dvou NOR perceptronů s křížovou zpětnou vazbou (stav drží smyčka, ne atribut)."""

    def __init__(self, g: Gates):
        self.g = g
        self.Q, self.Qn = 0, 1

    def step(self, S: int, R: int) -> int:
        for _ in range(8):                              # ustálení zpětné vazby
            q = self.g.run("NOR", R, self.Qn)
            qn = self.g.run("NOR", S, q)
            if (q, qn) == (self.Q, self.Qn):
                break
            self.Q, self.Qn = q, qn
        return self.Q


# ----------------------------------------------------------------- samokontrola
def _selfcheck() -> int:
    errors = []

    def check(ok: bool, msg: str):
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            errors.append(msg)

    g = Gates()
    for name, y in TRUTH.items():
        out = [g.run(name, a, b) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
        p = g.neurons[name]
        check(out == y, f"{name}: {out}  W={np.round(p.W, 2).tolist()} b={p.b:.2f}")

    check([g.xor(a, b) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))] == [0, 1, 1, 0], "XOR složením 3 neuronů")

    single = Perceptron(2, seed=0)
    single.fit(_X2, [0, 1, 1, 0], max_epochs=5000)
    check(not np.array_equal(single.predict(_X2), [0, 1, 1, 0]), "samotný neuron XOR nezvládne (očekávané)")

    bad = sum(1 for x in range(16) for y in range(16) if add(g, x, y) != ((x + y) & 15, (x + y) >> 4))
    check(bad == 0, f"ADD 4 bity: 256 kombinací, chyb {bad}")
    bad = sum(1 for x in range(16) for y in range(16) if sub(g, x, y) != ((x - y) & 15, int(x < y)))
    check(bad == 0, f"SUB 4 bity: 256 kombinací, chyb {bad}")

    latch = SRLatch(g)
    seq = [latch.step(*sr) for sr in ((1, 0), (0, 0), (0, 0), (0, 1), (0, 0))]
    check(seq == [1, 1, 1, 0, 0], f"SR latch SET/HOLD/RESET/HOLD: {seq}")

    print("VÝSLEDEK:", "OK" if not errors else f"{len(errors)} CHYB")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(_selfcheck())
