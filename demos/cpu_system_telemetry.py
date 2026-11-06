"""
cpu_system_telemetry.py - Komplexní telemetrie celého CPU.
Sleduje tok instrukcí, stav registrů, sběrnici RAM a vnitřní signály ALU při reálném programu.
"""

import os
import sys
import time
import numpy as np

from neuron import neuron_node as nn
from cpu.cpu_telemetry_probe import HardwareLogicProbe, ProbedNeuronALU
from cpu.cpu_step3_registers import NeuronRegister

class TelemetricSoftCPU:
    def __init__(self, probe: HardwareLogicProbe):
        self.probe = probe
        self.socket = nn.NeuronCore(input_dim=2, learning_rate=1.5, seed=42)
        self.alu = ProbedNeuronALU(probe)
        
        # NOR kalibrace pro paměť
        X_nor = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
        y_nor = np.array([1, 0, 0, 0])
        self.socket.train_fast(X_nor, y_nor, max_cas_sekund=0.5, target_loss=1e-3, batch_epochs=100)
        self.nor_W = self.socket.W.copy()
        self.nor_b = float(self.socket.b)

        self.ACC = NeuronRegister(num_bits=4, core=self.socket, nor_weights=self.nor_W, nor_bias=self.nor_b)
        self.ACC.write(0)
        self.PC = 0
        self.RAM = [0] * 16
        self.halted = False
        self.cycle = 0

    def load_program(self, program: list[tuple[int, int]]):
        for i, (op, oper) in enumerate(program):
            self.RAM[i*2] = op
            self.RAM[i*2+1] = oper

    def execute_cycle(self):
        if self.halted:
            return False
        
        self.cycle += 1
        opcode = self.RAM[self.PC]
        operand = self.RAM[self.PC + 1]
        pc_curr = self.PC
        self.PC += 2

        print(f"\n>>> [TAKT #{self.cycle:02d}] FETCH: PC=0x{pc_curr:X} -> OPCODE=0x{opcode:X}, OPERAND=0x{operand:X}")

        if opcode == 0x1: # LOAD
            val = self.RAM[operand]
            print(f"    DECODE: LOAD [0x{operand:X}]")
            print(f"    DATA-BUS: RAM[0x{operand:X}] vysílá hodnotu {val} (0b{bin(val)[2:].zfill(4)}) na sběrnici")
            self.ACC.write(val)
            print(f"    REG-BUS:  ACC nastaven na {self.ACC.read()}")

        elif opcode == 0x2: # ADD
            acc_val = self.ACC.read()
            ram_val = self.RAM[operand]
            print(f"    DECODE: ADD [0x{operand:X}]")
            print(f"    ALU-IN: A={acc_val} (ACC), B={ram_val} (RAM[0x{operand:X}])")
            
            # Sčítáme nultý bit
            s0, c0 = self.alu.half_adder_probed(acc_val & 1, ram_val & 1)
            # Sčítáme první bit s přenosem
            s1_raw, c1_a = self.alu.half_adder_probed((acc_val >> 1) & 1, (ram_val >> 1) & 1)
            s1, c1_b = self.alu.half_adder_probed(s1_raw, c0)
            cout = self.alu.eval_probed("ALU_Bit1_Cout", "OR", c1_a, c1_b)
            
            res_val = s0 | (s1 << 1)
            print(f"    ALU-OUT: Sum={res_val} (0b{s1}{s0}), Carry={cout}")
            self.ACC.write(res_val)
            print(f"    REG-BUS: ACC zapsán na {self.ACC.read()}")

        elif opcode == 0x3: # STORE
            acc_val = self.ACC.read()
            print(f"    DECODE: STORE [0x{operand:X}]")
            print(f"    DATA-BUS: ACC ({acc_val}) zapisuje do RAM[0x{operand:X}]")
            self.RAM[operand] = acc_val

        elif opcode == 0xF: # HALT
            print("    DECODE: HALT - Procesor zastaven.")
            self.halted = True

        return not self.halted

def run():
    probe = HardwareLogicProbe()
    cpu = TelemetricSoftCPU(probe)

    # Vložíme čísla: 1 a 2
    cpu.RAM[0xA] = 1
    cpu.RAM[0xB] = 2

    # Program: LOAD 0xA, ADD 0xB, STORE 0xC, HALT
    cpu.load_program([
        (0x1, 0xA),
        (0x2, 0xB),
        (0x3, 0xC),
        (0xF, 0x0)
    ])

    print("=" * 115)
    print("  START HLOUBKOVÉ TELEMETRIE CELÉHO PROCESORU (CYKLY & ODOSLECH SBĚRNICE)")
    print("=" * 115)

    while cpu.execute_cycle():
        pass

    # A teď vytiskneme osciloskopickou mapu všech hradel z taktu ADD!
    probe.dump_oscilloscope_view()

if __name__ == "__main__":
    run()
