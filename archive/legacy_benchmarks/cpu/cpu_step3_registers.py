"""
cpu_step3_registers.py - Krok 3 stavby Softwarového CPU:
Paměťové registry a bistabilní klopný obvod (SR-Latch / RAM bit).

Princip:
  - Bez paměti je neuron jen kombinatorický obvod (okamžitá reakce bez minulosti).
  - Přidáním zpětné vazby z L1 bufferu (výstup se vrací jako vstup pro další takt)
    vzniká bistabilní klopný obvod (SR-Latch):
      S = Set (nastav 1)
      R = Reset (nastav 0)
      Q = uchovávaný stav (0 nebo 1 i po odpojení S a R signálu!)
  - Seskupením do registrů získáme:
      - 4-bit / 8-bit Akumulátor (ACC)
      - Program Counter (PC - čítač instrukcí)
"""

import os
import sys
import numpy as np

from neuron import neuron_node as nn

class NeuronSRLatch:
    """
    1-bitový paměťový bistabilní obvod postavený ze dvou NOR bran s křížovou zpětnou vazbou.
    """
    def __init__(self, core: nn.NeuronCore, nor_weights: np.ndarray, nor_bias: float):
        self.core = core
        self.W = nor_weights
        self.b = nor_bias
        # Počáteční stav paměti Q=0, Q_bar=1
        self.Q = 0
        self.Q_bar = 1

    def _eval_nor(self, x1: int, x2: int) -> int:
        self.core.W = self.W
        self.core.b = self.b
        vec = np.array([[float(x1), float(x2)]])
        _, dec = self.core.predict(vec)
        return int(dec[0])

    def clock_tick(self, S: int, R: int) -> int:
        """
        Jeden hodinový takt paměťové buňky:
        S=1, R=0 -> Q = 1 (SET)
        S=0, R=1 -> Q = 0 (RESET)
        S=0, R=0 -> Q = Q_previous (HOLD / PAMĚŤ ZŮSTÁVÁ!)
        """
        # Obvod se ustálí (iterace zpětné vazby v L1 bufferu)
        for _ in range(3):
            self.Q_bar = self._eval_nor(S, self.Q)
            self.Q     = self._eval_nor(R, self.Q_bar)
        return self.Q


class NeuronRegister:
    """
    N-bitový procesorový registr (např. Akumulátor nebo Program Counter).
    Složený z pole neuronových Latch buněk.
    """
    def __init__(self, num_bits: int, core: nn.NeuronCore, nor_weights: np.ndarray, nor_bias: float):
        self.num_bits = num_bits
        self.cells = [NeuronSRLatch(core, nor_weights, nor_bias) for _ in range(num_bits)]

    def write(self, value: int):
        """Zápis celého čísla do registru (vygeneruje S a R pulzy pro každý bit)."""
        for i in range(self.num_bits):
            target_bit = (value >> i) & 1
            if target_bit == 1:
                self.cells[i].clock_tick(S=1, R=0) # SET
            else:
                self.cells[i].clock_tick(S=0, R=1) # RESET
            # Návrat na klidový stav (HOLD)
            self.cells[i].clock_tick(S=0, R=0)

    def read(self) -> int:
        """Přečtení uloženého čísla (čtení paměti při S=0, R=0)."""
        val = 0
        for i in range(self.num_bits):
            bit = self.cells[i].clock_tick(S=0, R=0)
            val |= (bit << i)
        return val


def run_step3_registers(verbose: bool = True):
    if verbose:
        print("\n" + "=" * 70)
        print("  KROK 3: BISTABILNÍ PAMĚŤ (SR-LATCH) & N-BITOVÉ REGISTRY")
        print("=" * 70)

    # 1. Kalibrace NOR brány pro paměťový element
    socket = nn.NeuronCore(input_dim=2, learning_rate=1.5, seed=42)
    X_nor = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
    y_nor = np.array([1, 0, 0, 0]) # NOR = NOT(OR)
    
    socket.train_fast(X_nor, y_nor, max_cas_sekund=0.5, target_loss=1e-3, batch_epochs=100)
    nor_W = socket.W.copy()
    nor_b = float(socket.b)

    # 2. Test 1-bit SR-Latch
    if verbose:
        print("\n[ 1. TEST 1-BITOVÉHO PAMĚŤOVÉHO ELEMENTU (SR-LATCH) ]")
    latch = NeuronSRLatch(socket, nor_W, nor_b)
    
    # Krok A: Nastavíme 1 (SET)
    q = latch.clock_tick(S=1, R=0)
    assert q == 1, f"Chyba SET: očekáváno 1, získáno {q}"
    if verbose:
        print("  1. Impulz SET (S=1, R=0) -> Q = 1")

    # Krok B: Odpojíme impulsy (HOLD) -> Stav MUSÍ ZŮSTAT 1!
    for cycle in range(1, 4):
        q = latch.clock_tick(S=0, R=0)
        assert q == 1, f"Chyba HOLD: paměť ztratila stav v cyklu {cycle}!"
        if verbose:
            print(f"  2.{cycle} Impulz HOLD (S=0, R=0) -> Q = 1 (Uchováno v L1 feedbacku)")

    # Krok C: Vymažeme (RESET) -> Q = 0
    q = latch.clock_tick(S=0, R=1)
    assert q == 0, f"Chyba RESET: očekáváno 0, získáno {q}"
    if verbose:
        print("  3. Impulz RESET (S=0, R=1) -> Q = 0")

    # Krok D: Opět HOLD -> Stav MUSÍ ZŮSTAT 0!
    q = latch.clock_tick(S=0, R=0)
    assert q == 0, f"Chyba HOLD: paměť nezůstala na 0!"
    if verbose:
        print("  4. Impulz HOLD (S=0, R=0) -> Q = 0 (Uchována nula)")

    # 3. Test 4-bitového procesorového registru (Akumulátor)
    if verbose:
        print("\n[ 2. TEST 4-BITOVÉHO PROCESOROVÉHO REGISTRU (ACC) ]")
    acc = NeuronRegister(num_bits=4, core=socket, nor_weights=nor_W, nor_bias=nor_b)

    test_values = [5, 12, 3, 15, 0, 9]
    for val in test_values:
        acc.write(val)
        read_back = acc.read()
        if verbose:
            bin_str = bin(read_back)[2:].zfill(4)
            print(f"  Zápis: {val:2d} -> Čtení po odpojení sběrnice: {read_back:2d} (0b{bin_str})")
        assert read_back == val, f"Chyba registru: zapsáno {val}, přečteno {read_back}"

    if verbose:
        print("=" * 70)
        print("  >>> VERIFIKACE ÚSPĚŠNÁ: PAMĚŤOVÉ REGISTRY ZCELA FUNKČNÍ! (KROK 3 HOTOV) <<<")
        print("=" * 70 + "\n")
    return True

if __name__ == "__main__":
    if not run_step3_registers(verbose=True):
        sys.exit(1)
