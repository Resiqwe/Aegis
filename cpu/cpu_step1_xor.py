"""
cpu_step1_xor.py - Krok 1 stavby Softwarového CPU:
Syntéza XOR hradla z elementárních neuronových buněk přes L1 Transit Buffer.

Topologie (Univerzální baze):
  Vstup: X [4x2]
  Fáze A:
    - Node A1 (NAND brána): učí se NOT(x1 AND x2)
    - Node A2 (OR brána):   učí se (x1 OR x2)
  L1 Transit Buffer:
    - Shromáždí predikce [pred_NAND, pred_OR] do matice X_stageB [4x2]
  Fáze B:
    - Node B (AND brána):   učí se (pred_NAND AND pred_OR)
  Výsledek:
    - Striktní XOR: [0, 1, 1, 0]
"""

import os
import sys
import numpy as np

from neuron import neuron_node as nn

class L1TransitBuffer:
    """Procesorový L1 meziregistr v paměti RAM."""
    def __init__(self):
        self.data_store = {}

    def store(self, key: str, data: np.ndarray):
        self.data_store[key] = np.array(data, dtype=np.float64, copy=True)

    def fetch(self, key: str) -> np.ndarray:
        if key not in self.data_store:
            raise KeyError(f"L1 Transit Buffer: klíč '{key}' nebyl nalezen!")
        return self.data_store[key]

    def build_matrix(self, keys: list) -> np.ndarray:
        """Složí sloupce do nové matice pro další procesorovou fázi."""
        cols = [self.fetch(k) for k in keys]
        return np.column_stack(cols)


def run_step1_xor(verbose: bool = True):
    if verbose:
        print("\n" + "=" * 70)
        print("  KROK 1: SYNTÉZA XOR HRADLA PŘES L1 BUFFER (NELINEARITA)")
        print("=" * 70)

    # Pravdivostní tabulka pro 2 bity
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])

    y_nand = np.array([1, 1, 1, 0])          # NAND
    y_or   = np.array([0, 1, 1, 1])          # OR
    
    # Cílový XOR: [0, 1, 1, 0]
    # Všimněme si matice X_stageB:
    # vstup [0, 0] -> NAND=1, OR=0 -> chceme výstup 0
    # vstup [0, 1] -> NAND=1, OR=1 -> chceme výstup 1
    # vstup [1, 0] -> NAND=1, OR=1 -> chceme výstup 1
    # vstup [1, 1] -> NAND=0, OR=1 -> chceme výstup 0
    # Cíl pro Uzel B je tedy přesně: když NAND=1 a OR=1, tak 1, jinak 0 -> y = [0, 1, 1, 0]
    y_target_stageB = np.array([0, 1, 1, 0])
    y_xor_expected  = np.array([0, 1, 1, 0])  # Očekávaný cíl XOR

    l1 = L1TransitBuffer()

    # 1 fyzické výpočetní jádro v paměti (2 vstupy)
    socket = nn.NeuronCore(input_dim=2, learning_rate=1.5, seed=42)

    # -------------------------------------------------------------
    # FÁZE A: Paralelní uzly A1 (NAND) a A2 (OR)
    # -------------------------------------------------------------
    if verbose:
        print("\n[ FÁZE A: Výpočet primárních hradel na 1 hardware socketu ]")

    # A1: NAND
    stats_nand = socket.train_fast(X, y_nand, max_cas_sekund=1.0, target_loss=1e-3, batch_epochs=100)
    probs_nand, dec_nand = socket.predict(X)
    l1.store("nand_out", dec_nand)
    if verbose:
        print(f"  -> Uzel A1 (NAND): Výstup = {dec_nand.tolist()} | Loss = {stats_nand['mse_loss']:.2e} | Čas = {stats_nand['elapsed_ms']:.2f} ms")

    # Časový multiplexing: resetujeme váhy stejného socketu v RAM
    socket.W = np.random.randn(2)
    socket.b = 0.0

    # A2: OR
    stats_or = socket.train_fast(X, y_or, max_cas_sekund=1.0, target_loss=1e-3, batch_epochs=100)
    probs_or, dec_or = socket.predict(X)
    l1.store("or_out", dec_or)
    if verbose:
        print(f"  -> Uzel A2 (OR):   Výstup = {dec_or.tolist()} | Loss = {stats_or['mse_loss']:.2e} | Čas = {stats_or['elapsed_ms']:.2f} ms")

    # -------------------------------------------------------------
    # L1 TRANSIT BUFFER: Příprava vstupní matice pro Fázi B
    # -------------------------------------------------------------
    X_stageB = l1.build_matrix(["nand_out", "or_out"])
    if verbose:
        print("\n[ L1 BUFFER: Fúze signálů na sběrnici ]")
        print(f"  Matice pro Fázi B (NAND & OR sloupce):\n{X_stageB}")

    # -------------------------------------------------------------
    # FÁZE B: Finální syntéza v jádru (AND brána)
    # -------------------------------------------------------------
    if verbose:
        print("\n[ FÁZE B: Syntéza v jádru (AND brána) ]")

    socket.W = np.random.randn(2)
    socket.b = 0.0

    stats_xor = socket.train_fast(X_stageB, y_target_stageB, max_cas_sekund=1.0, target_loss=1e-3, batch_epochs=100)
    probs_xor, dec_xor = socket.predict(X_stageB)

    if verbose:
        print(f"  -> Uzel B (XOR Výstup): {dec_xor.tolist()} | Loss = {stats_xor['mse_loss']:.2e} | Čas = {stats_xor['elapsed_ms']:.2f} ms")
        print("=" * 70)

    # Ověření pravdivosti
    is_success = np.array_equal(dec_xor, y_xor_expected)
    if is_success:
        if verbose:
            print("  >>> VERIFIKACE ÚSPĚŠNÁ: XOR ZCELA VYŘEŠEN! (KROK 1 HOTOV) <<<")
            print("=" * 70 + "\n")
        return True, dec_xor
    else:
        if verbose:
            print("  >>> CHYBA: Výstup neodpovídá XOR! <<<")
            print("=" * 70 + "\n")
        return False, dec_xor

if __name__ == "__main__":
    success, out = run_step1_xor(verbose=True)
    if not success:
        sys.exit(1)
