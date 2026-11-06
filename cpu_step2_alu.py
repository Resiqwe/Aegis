"""
cpu_step2_alu.py - Krok 2 stavby Softwarového CPU:
Aritmeticko-logická jednotka (ALU) - Half-Adder a Full-Adder.

Princip:
  - Half-Adder sčítá 2 bity: A + B
      Sum   = A XOR B
      Carry = A AND B
  - Full-Adder sčítá 3 bity: A + B + Cin
      Sum     = (A XOR B) XOR Cin
      Cout    = (A AND B) OR (Cin AND (A XOR B))
  - 2-bit Adder: Sčítá dvě 2-bitová čísla (např. 2 + 3 = 5, resp. 0b10 + 0b11 = 0b101)
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(__file__))
import neuron_node as nn

class NeuronALU:
    """
    ALU postavená z elementárních neuronových buněk a L1 sběrnice.
    """
    def __init__(self):
        # 1 fyzické výpočetní jádro v paměti pro časový multiplexing
        self.socket = nn.NeuronCore(input_dim=2, learning_rate=1.5, seed=42)
        # Natrénované váhy pro elementární hradla (ROM paměť mikrokódu ALU)
        self.microcode_rom = {}
        self._calibrate_gates()

    def _calibrate_gates(self):
        """Předkalibrace základních hradel (ROM pro instrukce ALU)."""
        X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
        targets = {
            "AND":  np.array([0, 0, 0, 1]),
            "OR":   np.array([0, 1, 1, 1]),
            "NAND": np.array([1, 1, 1, 0])
        }
        for gate, y in targets.items():
            self.socket.W = np.random.randn(2)
            self.socket.b = 0.0
            self.socket.train_fast(X, y, max_cas_sekund=0.5, target_loss=1e-3, batch_epochs=100)
            self.microcode_rom[gate] = (self.socket.W.copy(), float(self.socket.b))

    def eval_gate(self, gate_name: str, a: int, b: int) -> int:
        """Vyhodnocení hradla na socketu pomocí mikrokódu z ROM."""
        W, bias = self.microcode_rom[gate_name]
        self.socket.W = W
        self.socket.b = bias
        vec = np.array([[float(a), float(b)]])
        _, dec = self.socket.predict(vec)
        return int(dec[0])

    def xor(self, a: int, b: int) -> int:
        """Složený XOR z L1 sběrnice: NAND(a,b) AND OR(a,b)."""
        nand_out = self.eval_gate("NAND", a, b)
        or_out   = self.eval_gate("OR", a, b)
        # Syntéza AND
        return self.eval_gate("AND", nand_out, or_out)

    def half_adder(self, a: int, b: int) -> tuple[int, int]:
        """
        1-bit Half-Adder:
        Vrací (Sum, Carry)
        """
        sum_bit = self.xor(a, b)
        carry_bit = self.eval_gate("AND", a, b)
        return sum_bit, carry_bit

    def full_adder(self, a: int, b: int, cin: int) -> tuple[int, int]:
        """
        1-bit Full-Adder:
        Vrací (Sum, Cout)
        """
        s1, c1 = self.half_adder(a, b)
        sum_bit, c2 = self.half_adder(s1, cin)
        cout = self.eval_gate("OR", c1, c2)
        return sum_bit, cout

    def add_2bit(self, a_bits: list[int], b_bits: list[int]) -> list[int]:
        """
        2-bitová paralelní sčítačka.
        a_bits: [A0 (LSB), A1 (MSB)]
        b_bits: [B0 (LSB), B1 (MSB)]
        Vrací: [S0, S1, Cout] (3 bity)
        """
        # Nultý bit (LSB) sčítáme přes Half-Adder
        s0, c0 = self.half_adder(a_bits[0], b_bits[0])
        # První bit sčítáme přes Full-Adder s přenosem c0
        s1, cout = self.full_adder(a_bits[1], b_bits[1], c0)
        return [s0, s1, cout]

def run_step2_alu(verbose: bool = True):
    if verbose:
        print("\n" + "=" * 70)
        print("  KROK 2: ARITMETICKO-LOGICKÁ JEDNOTKA (ALU) & SČÍTAČKA")
        print("=" * 70)

    alu = NeuronALU()

    if verbose:
        print("\n[ 1. TEST HALF-ADDERU (1-bit bez přenosu) ]")
        print("  A + B -> (Sum, Carry)")
    
    ha_cases = [
        (0, 0, (0, 0)),
        (0, 1, (1, 0)),
        (1, 0, (1, 0)),
        (1, 1, (0, 1))
    ]
    for a, b, expected in ha_cases:
        res = alu.half_adder(a, b)
        if verbose:
            print(f"  {a} + {b} = Sum: {res[0]}, Carry: {res[1]} (Očekáváno: {expected})")
        assert res == expected, f"Chyba Half-Adder: {a}+{b} dalo {res}"

    if verbose:
        print("\n[ 2. TEST 2-BITOVÉ PARALELNÍ SČÍTAČKY (Všech 16 kombinací) ]")
        print("  A (0..3) + B (0..3) = Výsledek (0..6)")

    total_correct = 0
    for a_val in range(4):
        for b_val in range(4):
            a_bits = [a_val & 1, (a_val >> 1) & 1]
            b_bits = [b_val & 1, (b_val >> 1) & 1]
            out_bits = alu.add_2bit(a_bits, b_bits)
            res_val = out_bits[0] + (out_bits[1] << 1) + (out_bits[2] << 2)
            expected_val = a_val + b_val
            assert res_val == expected_val, f"Selhalo: {a_val} + {b_val} = {res_val} (očekáváno {expected_val})"
            total_correct += 1

    if verbose:
        print(f"  Ověřeno všech {total_correct}/16 binárních sčítání!")
        print("  Příklady:")
        print("    1 + 1 = 2 (0b01 + 0b01 = 0b010)")
        print("    2 + 3 = 5 (0b10 + 0b11 = 0b101)")
        print("    3 + 3 = 6 (0b11 + 0b11 = 0b110)")
        print("=" * 70)
        print("  >>> VERIFIKACE ÚSPĚŠNÁ: ARITMETICKÁ ALU PLNE FUNKČNÍ! (KROK 2 HOTOV) <<<")
        print("=" * 70 + "\n")
    return True

if __name__ == "__main__":
    if not run_step2_alu(verbose=True):
        sys.exit(1)
