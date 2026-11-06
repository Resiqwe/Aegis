"""
CHAIN RUNNER (ORCHESTRÁTOR ŘETĚZENÍ A L1 TRANSIT BUFFER)
=========================================================
Architektura:
- 1 fyzický výpočetní socket v paměti (NeuronCore neopouští RAM)
- L1 Transit Buffer: drží mezistav mezi takty a propojuje buňky
- Časový multiplexing (Fáze A -> Transit Buffer -> Reset socketu -> Fáze B -> Zapečetěný výstup)
"""
import time
import json
import numpy as np
from pathlib import Path

import neuron_node as nn

BASE_DIR = Path(__file__).resolve().parent
DUMP_DIR = nn.DUMP_DIR
LOG_DIR = nn.LOG_DIR


class L1TransitBuffer:
    """
    Simulace L1 vyrovnávací paměti řadiče.
    Uchovává aktivační potenciály (pravděpodobnosti) z předchozího taktu
    a formátuje je jako vstupní tenzor pro další takt.
    """
    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.slots = []

    def push(self, data: list[float] | np.ndarray, tag: str = "TICK_OUT"):
        """Uloží výstup do bufferu."""
        data_arr = np.array(data, dtype=np.float64)
        self.slots.append({"tag": tag, "data": data_arr, "timestamp_us": time.time_ns() // 1000})
        if len(self.slots) > self.capacity:
            self.slots.pop(0)

    def pull_as_matrix(self) -> np.ndarray:
        """
        Vyzvedne uložená data z bufferu a zformátuje je jako matici vstupů pro další fázi.
        Každý prvek z předchozího výstupu namapuje do vstupního vektoru: [p, p^2, 1-p]
        """
        if not self.slots:
            raise RuntimeError("[L1 BUFFER] Pokus o čtení z prázdného bufferu!")
        latest = self.slots[-1]["data"]
        X_next = np.column_stack([
            latest,
            latest ** 2,
            1.0 - latest
        ])
        return X_next

    def clear(self):
        self.slots.clear()


class ChainController:
    """
    Hlavní řadič: drží v paměti neuronový socket a přepíná fáze řetězu.
    """
    def __init__(self):
        print("  [ŘADIČ] Inicializace hardware socketu v paměti...")
        # 1 fyzické jádro alokované v paměti
        self.socket = nn.NeuronCore(input_dim=3, learning_rate=0.5, seed=42)
        self.l1_buffer = L1TransitBuffer(capacity=8)

    def verify_stage(self, meta_path: Path, X_stage: np.ndarray, y_target: np.ndarray):
        """Zpětná kontrola specifická pro vstupní data dané fáze."""
        with open(meta_path, "r", encoding="utf-8-sig") as f:
            meta = json.load(f)
        weights_path = DUMP_DIR / meta_path.name.replace("_meta.json", "_weights.dat")
        with open(weights_path, "r", encoding="utf-8-sig") as f:
            params = [float(x) for x in f.read().split()]

        temp_core = nn.NeuronCore(input_dim=len(params) - 1)
        temp_core.W = np.array(params[:-1], dtype=np.float64)
        temp_core.b = float(params[-1])
        probs, decision = temp_core.predict(X_stage)

        if not np.allclose(probs, meta["probabilities"], atol=1e-9):
            raise RuntimeError(f"Fáze {meta_path.name}: predikce z disku nesedí s občankou!")
        if not np.array_equal(decision, y_target):
            raise RuntimeError(f"Fáze {meta_path.name}: rozhodnutí nesedí s cílem!")
        print(f"  [VERIFIKACE] {meta_path.name} dává z disku identickou predikci a trefuje cíl 100%.")

    def run_twostage_chain(self, X_input: np.ndarray, y_target: np.ndarray):
        print("\n" + "=" * 80)
        print("  SPUŠTĚNÍ ŘETĚZOVÉHO BĚHU (ČASOVÝ MULTIPLEXING NA 1 JÁDŘE)")
        print("=" * 80)

        # ----------------------------------------------------------------------
        # FÁZE A: Primární zpracování surových dat
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        print("\n>>> [FÁZE A] Jádro běží na primárních datech (Socket stav: RAW)...")
        stats_a = self.socket.train_fast(
            X_input, y_target, max_cas_sekund=1.0, target_loss=1e-4, batch_epochs=100
        )
        meta_a, hash_a = nn.extract_and_dump(self.socket, stats_a, run_id=1, silent=True)
        print(f"    -> Fáze A dokončena: {stats_a['elapsed_ms']:.2f} ms | Loss: {stats_a['mse_loss']:.2e}")
        print(f"    -> Výstupní pravděpodobnosti Fáze A: {[round(p, 4) for p in stats_a['probabilities']]}")

        # FLUSNUTÍ DO L1 BUFFERU
        self.l1_buffer.push(stats_a['probabilities'], tag="PHASE_A_OUT")
        print("    -> Výstupy Fáze A bezpečně uloženy do L1 Transit Bufferu.")

        # ----------------------------------------------------------------------
        # MEZIDOBÍ: Reset a rekonfigurace jádra (Jádro NEOPUŠTĚLO paměť!)
        # ----------------------------------------------------------------------
        print("\n--- [ŘADIČ: PŘEPNUTÍ SBĚRNICE] Reset vah jádra v RAM (Zero-realloc) ---")
        np.random.seed(1337)  # Nový kontext pro druhý stupeň
        self.socket.W = np.random.randn(3)
        self.socket.b = 0.0
        self.socket.total_epochs = 0

        # Příprava vstupů z L1 bufferu pro Fázi B
        X_from_l1 = self.l1_buffer.pull_as_matrix()
        print(f"    -> L1 Buffer vygeneroval novou vstupní matici pro Fázi B:\n{np.round(X_from_l1, 3)}")

        # ----------------------------------------------------------------------
        # FÁZE B: Sekundární zpracování dat z L1 bufferu
        # ----------------------------------------------------------------------
        print("\n>>> [FÁZE B] To samé jádro nyní zpracovává data z L1 Transit Bufferu...")
        stats_b = self.socket.train_fast(
            X_from_l1, y_target, max_cas_sekund=1.0, target_loss=5e-5, batch_epochs=100
        )
        meta_b, hash_b = nn.extract_and_dump(
            self.socket, stats_b, run_id=2, parent_hash=hash_a, silent=True
        )
        print(f"    -> Fáze B dokončena: {stats_b['elapsed_ms']:.2f} ms | Loss: {stats_b['mse_loss']:.2e}")
        print(f"    -> Finální rozhodnutí řetězce: {[round(p, 4) for p in stats_b['probabilities']]}")

        total_chain_ms = (time.perf_counter() - t0) * 1000.0
        print("\n" + "=" * 80)
        print(f"  [ŘETĚZ DOKONČEN] Obě fáze proběhly na 1 jádře za celkem {total_chain_ms:.2f} ms")
        print(f"  Rodokmen řetězu: Fáze A ({hash_a[:8]}) ===[L1 BUFFER]===> Fáze B ({hash_b[:8]})")
        print("=" * 80)

        # Kontrola občanek
        nn.validate_dumps([meta_a, meta_b], quiet=False)

        # Zpětná kontrola každé fáze proti jejím vlastním vstupům
        self.verify_stage(meta_a, X_input, y_target)
        self.verify_stage(meta_b, X_from_l1, y_target)


if __name__ == "__main__":
    controller = ChainController()
    controller.run_twostage_chain(nn.X_env, nn.y_env)
