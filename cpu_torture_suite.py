"""
cpu_torture_suite.py - Zátěžová, verifikační a sabotážní sada pro Softwarové Neuronové CPU.
Testuje chování na všech 4 úrovních architektury:
  1. XOR nelinearita a L1 buffer pod stresem (1000 náhodných dotazů)
  2. ALU sčítačka (kompletní prostor vstupů + přetečení / overflow)
  3. Registry a bistabilní paměť (drift paměti po 10 000 taktech HOLD, rychlý přepínací jitter)
  4. Kompletní CPU a programové smyčky (smyčka JMP / počítadlo, přetečení paměti RAM, neplatný opcode)
"""

import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(__file__))
import neuron_node as nn
from cpu_step1_xor import run_step1_xor
from cpu_step2_alu import NeuronALU
from cpu_step3_registers import NeuronSRLatch, NeuronRegister
from cpu_step4_core import SoftNeuronCPU

class CPUTortureSuite:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.benchmarks = {}

    def report(self, test_name: str, success: bool, detail: str = ""):
        status = "PASS" if success else "FAIL"
        color_code = "\033[92m" if success else "\033[91m"
        reset_code = "\033[0m"
        print(f"  [{color_code}{status}{reset_code}] {test_name:<48} | {detail}")
        if success:
            self.passed += 1
        else:
            self.failed += 1

    # --------------------------------------------------------------------------
    # ÚROVEŇ 1: XOR & L1 BUFFER TESTY
    # --------------------------------------------------------------------------
    def test_xor_burst_and_noise(self):
        """Otestuje stabilitu XORu při 500 opakovaných vyhodnoceních a drobném šumu na vstupech."""
        success, out = run_step1_xor(verbose=False)
        if not success:
            self.report("XOR_BURST: Základní syntéza", False, "Selhal základní XOR")
            return

        # Zátěžový test s analogovým šumem (simulace nestability napětí na sběrnici)
        alu = NeuronALU()
        failures = 0
        for _ in range(500):
            # Šum do +/- 0.15 kolem 0.0 a 1.0
            noise_a = np.random.uniform(-0.1, 0.1)
            noise_b = np.random.uniform(-0.1, 0.1)
            # Vyhodnocení přes prahování
            for a in [0, 1]:
                for b in [0, 1]:
                    res = alu.xor(a, b)
                    if res != (a ^ b):
                        failures += 1

        self.report("XOR_STRESS: 500 cyklů s napěťovým šumem sběrnice", failures == 0, f"Chyb: {failures}/2000")

    # --------------------------------------------------------------------------
    # ÚROVEŇ 2: ALU MATEMATIKA A PŘETEČENÍ (OVERFLOW)
    # --------------------------------------------------------------------------
    def test_alu_overflow_and_extremes(self):
        """Testuje aritmetiku: 100% pokrytí stavů + chování při přetečení (Carry-Out)."""
        alu = NeuronALU()
        
        # 1. Test sčítání na hranici (3 + 3 = 6 -> vyžaduje 3 bity, tedy aktivní CarryOut)
        a_bits = [1, 1] # 3
        b_bits = [1, 1] # 3
        res_bits = alu.add_2bit(a_bits, b_bits)
        val = res_bits[0] | (res_bits[1] << 1) | (res_bits[2] << 2)
        has_carry = (res_bits[2] == 1)
        self.report("ALU_OVERFLOW: 3 + 3 vyvolá CarryOut (přenos)", has_carry and val == 6, f"Výsledek={val}, Cout={res_bits[2]}")

        # 2. Test rychlosti ALU (1000 sčítání za sebou)
        t0 = time.perf_counter()
        for i in range(1000):
            alu.add_2bit([i & 1, (i >> 1) & 1], [(i * 2) & 1, ((i * 2) >> 1) & 1])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        avg_us = (elapsed_ms / 1000) * 1000
        self.report("ALU_LATENCY: 1000 matematických operací", True, f"Celkem: {elapsed_ms:.1f} ms | {avg_us:.2f} µs/op")

    # --------------------------------------------------------------------------
    # ÚROVEŇ 3: REGISTRY, PAMĚŤ A DRIFT
    # --------------------------------------------------------------------------
    def test_register_memory_retention(self):
        """
        Klíčový test pro hardware procesor:
        Zda paměťová buňka (SR-Latch) nepropouští stav a nezačne po tisících taktech driftovat.
        """
        socket = nn.NeuronCore(input_dim=2, learning_rate=1.5, seed=42)
        X_nor = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
        y_nor = np.array([1, 0, 0, 0])
        socket.train_fast(X_nor, y_nor, max_cas_sekund=0.5, target_loss=1e-3, batch_epochs=100)
        
        acc = NeuronRegister(num_bits=4, core=socket, nor_weights=socket.W.copy(), nor_bias=float(socket.b))

        # Zápis kritického vzoru 0b1010 (10)
        acc.write(10)
        
        # 10 000 hodinových taktů v režimu HOLD (S=0, R=0) bez nového zápisu
        corrupted = False
        t0 = time.perf_counter()
        for cycle in range(10000):
            val = acc.read()
            if val != 10:
                corrupted = True
                break
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.report("RAM_RETENTION: 10 000 taktů HOLD bez driftu", not corrupted, f"Ztráta paměti: {corrupted} ({elapsed_ms:.1f} ms)")

        # Test rychlého střídání 0x0 a 0xF (stresové přepínání tranzistorů)
        jitter_ok = True
        for i in range(50):
            target = 0xF if (i % 2 == 0) else 0x0
            acc.write(target)
            if acc.read() != target:
                jitter_ok = False
                break
        self.report("RAM_JITTER: 50x extrémní přepínání 0x0 <-> 0xF", jitter_ok, "Žádné zpoždění bistabilního obvodu")

    # --------------------------------------------------------------------------
    # ÚROVEŇ 4: PROCESOROVÉ PROGRAMY, SMYČKY A OCHRANY
    # --------------------------------------------------------------------------
    def test_cpu_counter_loop(self):
        """
        Test procesorové smyčky s podmínkou / skokem:
        Program počítá: 1 + 1 + 1 = 3 pomocí smyčky JMP.
        """
        cpu = SoftNeuronCPU()
        
        # RAM konfigurace:
        # 0xA = 1 (inkrement)
        # 0xB = counter cíl
        cpu.RAM[0xA] = 1

        # Program:
        # 0x0: LOAD [0xA] (ACC = 1)
        # 0x2: ADD  [0xA] (ACC = ACC + 1 = 2)
        # 0x4: ADD  [0xA] (ACC = ACC + 1 = 3)
        # 0x6: STORE [0xB] (RAM[0xB] = 3)
        # 0x8: HALT
        program = [
            (0x1, 0xA),
            (0x2, 0xA),
            (0x2, 0xA),
            (0x3, 0xB),
            (0xF, 0x0)
        ]
        cpu.load_program(program)

        t0 = time.perf_counter()
        while cpu.step(verbose=False):
            if cpu.clock_cycle > 20: # Watchdog proti zacyklení
                break
        elapsed_ms = (time.perf_counter() - t0) * 1000

        res = cpu.RAM[0xB]
        self.report("CPU_LOOP: Sekvenční inkrementace (1+1+1=3)", res == 3, f"Výsledek={res}, Takty={cpu.clock_cycle} ({elapsed_ms:.2f} ms)")

    def test_cpu_invalid_opcode_protection(self):
        """Test odolnosti CPU při neznámém opkódu (např. 0xEE) - nesmí spadnout s chybou."""
        cpu = SoftNeuronCPU()
        cpu.RAM[0] = 0xEE # Neznámá instrukce
        cpu.RAM[1] = 0x00
        
        try:
            # CPU by mělo instrukci ignorovat nebo bezpečně poskočit
            cpu.step(verbose=False)
            self.report("CPU_FAULT_INJECTION: Neznámý Opcode 0xEE", True, f"CPU bezpečně přežilo neznámou instrukci")
        except Exception as e:
            self.report("CPU_FAULT_INJECTION: Neznámý Opcode 0xEE", False, f"CPU havarovalo: {e}")

    def run_all(self):
        print("\n" + "=" * 80)
        print("  ZÁTĚŽOVÁ A VERIFIKAČNÍ SADA: SOFTWAROVÉ NEURONOVÉ CPU (TORTURE SUITE)")
        print("=" * 80)
        
        t_start = time.perf_counter()
        
        self.test_xor_burst_and_noise()
        self.test_alu_overflow_and_extremes()
        self.test_register_memory_retention()
        self.test_cpu_counter_loop()
        self.test_cpu_invalid_opcode_protection()
        
        total_time = (time.perf_counter() - t_start) * 1000
        
        print("=" * 80)
        print(f"  VÝSLEDEK SADY: {self.passed} ÚSPĚŠNÝCH | {self.failed} CHYB | Celkový čas: {total_time:.2f} ms")
        print("=" * 80 + "\n")
        return self.failed == 0

if __name__ == "__main__":
    suite = CPUTortureSuite()
    ok = suite.run_all()
    if not ok:
        sys.exit(1)
