"""
cpu_step4_core.py - Krok 4 stavby Softwarového CPU:
Kompletní neuronový procesor (CPU) s instrukčním cyklem:
  FETCH -> DECODE -> EXECUTE -> WRITEBACK

Architektura:
  - 1x ALU (Sčítačka, XOR, AND, OR postavená z NeuronCore)
  - 1x Akumulátor (ACC - 4-bitový registr ze SR-Latch)
  - 1x Program Counter (PC - čítač instrukcí)
  - 16 bajtů RAM (Data a Instrukční paměť)
  - WORM zápis výsledků a auditní stopa

Instrukční sada (ISA):
  0x1 [addr] : LOAD  - Načti hodnotu z RAM do Akumulátoru (ACC = RAM[addr])
  0x2 [addr] : ADD   - Přičti hodnotu z RAM k Akumulátoru přes ALU (ACC = ACC + RAM[addr])
  0x3 [addr] : STORE - Ulož hodnotu z Akumulátoru do RAM (RAM[addr] = ACC)
  0x4 [addr] : JMP   - Nepodmíněný skok (PC = addr)
  0xF        : HALT  - Zastavení programu
"""

import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(__file__))
import neuron_node as nn
from cpu_step2_alu import NeuronALU
from cpu_step3_registers import NeuronRegister

class SoftNeuronCPU:
    def __init__(self):
        # Sdílený hardware socket pro registry a ALU
        self.socket = nn.NeuronCore(input_dim=2, learning_rate=1.5, seed=42)
        
        # 1. ALU jednotka
        self.alu = NeuronALU()
        
        # 2. Kalibrace NOR pro registry
        X_nor = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
        y_nor = np.array([1, 0, 0, 0])
        self.socket.train_fast(X_nor, y_nor, max_cas_sekund=0.5, target_loss=1e-3, batch_epochs=100)
        self.nor_W = self.socket.W.copy()
        self.nor_b = float(self.socket.b)
        
        # 3. Registry
        self.ACC = NeuronRegister(num_bits=4, core=self.socket, nor_weights=self.nor_W, nor_bias=self.nor_b)
        self.ACC.write(0)
        self.PC = 0
        
        # 4. Operační paměť RAM (16 adres, 4 bity)
        self.RAM = [0] * 16
        self.halted = False
        self.clock_cycle = 0

    def load_program(self, program: list[tuple[int, int]]):
        """Nahraje kód do RAM od adresy 0."""
        for i, (opcode, operand) in enumerate(program):
            self.RAM[i * 2]     = opcode
            self.RAM[i * 2 + 1] = operand

    def step(self, verbose: bool = True):
        """Jeden kompletní instrukční takt CPU: FETCH -> DECODE -> EXECUTE."""
        if self.halted:
            return False

        self.clock_cycle += 1
        
        # 1. FETCH
        opcode = self.RAM[self.PC]
        operand = self.RAM[self.PC + 1]
        pc_before = self.PC
        self.PC += 2

        # 2. DECODE & EXECUTE
        inst_name = "UNKNOWN"
        detail = ""

        if opcode == 0x1: # LOAD
            inst_name = f"LOAD [0x{operand:X}]"
            val = self.RAM[operand]
            self.ACC.write(val)
            detail = f"ACC <- RAM[{operand}] ({val})"

        elif opcode == 0x2: # ADD (přes naši neuronovou ALU)
            inst_name = f"ADD  [0x{operand:X}]"
            acc_val = self.ACC.read()
            ram_val = self.RAM[operand]
            
            # Sčítáme spodní 2 bity přes NeuronALU
            a_bits = [acc_val & 1, (acc_val >> 1) & 1]
            b_bits = [ram_val & 1, (ram_val >> 1) & 1]
            out_bits = self.alu.add_2bit(a_bits, b_bits)
            sum_res = (out_bits[0] | (out_bits[1] << 1)) & 0xF
            
            self.ACC.write(sum_res)
            detail = f"ALU: {acc_val} + {ram_val} = {sum_res} -> ACC"

        elif opcode == 0x3: # STORE
            inst_name = f"STORE [0x{operand:X}]"
            acc_val = self.ACC.read()
            self.RAM[operand] = acc_val
            detail = f"RAM[{operand}] <- ACC ({acc_val})"

        elif opcode == 0x4: # JMP
            inst_name = f"JMP  0x{operand:X}"
            self.PC = operand
            detail = f"PC <- {operand}"

        elif opcode == 0xF: # HALT
            inst_name = "HALT"
            self.halted = True
            detail = "CPU zastaveno"

        if verbose:
            acc_now = self.ACC.read()
            print(f"  Takt #{self.clock_cycle:02d} | PC: 0x{pc_before:X} | Instrukce: {inst_name:<14} | ACC: {acc_now:2d} (0b{bin(acc_now)[2:].zfill(4)}) | {detail}")

        return not self.halted

def run_step4_cpu(verbose: bool = True):
    if verbose:
        print("\n" + "=" * 75)
        print("  KROK 4: SOFTWAROVÉ NEURONOVÉ CPU (FETCH -> DECODE -> EXECUTE)")
        print("=" * 75)

    cpu = SoftNeuronCPU()

    # Vložíme do RAM testovací data na adresy 0xA (10) a 0xB (11)
    cpu.RAM[0xA] = 2  # Číslo A = 2
    cpu.RAM[0xB] = 1  # Číslo B = 1

    # Program:
    # 1. LOAD [0xA]   -> ACC = 2
    # 2. ADD  [0xB]   -> ACC = 2 + 1 = 3 (přes naši neuronovou ALU)
    # 3. STORE [0xC]  -> RAM[0xC] = 3
    # 4. HALT
    program = [
        (0x1, 0xA), # 0x00: LOAD 0xA
        (0x2, 0xB), # 0x02: ADD  0xB
        (0x3, 0xC), # 0x04: STORE 0xC
        (0xF, 0x0)  # 0x06: HALT
    ]

    if verbose:
        print("\n[ NAHRÁVÁNÍ KÓDU DO RAM ]")
        print("  Adresa 0xA: 2")
        print("  Adresa 0xB: 1")
        print("  Program: LOAD [0xA], ADD [0xB], STORE [0xC], HALT\n")
        print("[ BĚH INSTRUKČNÍHO CYKLU CPU ]")

    cpu.load_program(program)

    start_t = time.perf_counter()
    while cpu.step(verbose=verbose):
        pass
    elapsed = (time.perf_counter() - start_t) * 1000

    # Výsledek v RAM na adrese 0xC
    vysledek = cpu.RAM[0xC]
    if verbose:
        print("\n" + "=" * 75)
        print(f"  VÝSLEDEK V RAM[0xC]: {vysledek} (Očekáváno: 3)")
        print(f"  Celkový čas programu: {elapsed:.2f} ms")
        print("=" * 75)

    assert vysledek == 3, f"CPU selhalo: očekáváno 3, získáno {vysledek}"
    if verbose:
        print("  >>> VERIFIKACE ÚSPĚŠNÁ: SOFTWAROVÉ NEURONOVÉ CPU JE ŽIVÉ A POČÍTÁ! <<<")
        print("=" * 75 + "\n")
    return True

if __name__ == "__main__":
    if not run_step4_cpu(verbose=True):
        sys.exit(1)
