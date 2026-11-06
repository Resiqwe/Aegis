"""
cpu_hardware_clock_core.py - Pokročilé Softwarové CPU s fyzikálním taktovacím generátorem (Clock Generator)
a plnou Turingovskou výbavou (Zero Flag & Podmíněné skoky JNZ / JZ).
S podporou rozšířené paměti RAM (256 bajtů).
"""

import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(__file__))
import neuron_node as nn
from cpu_telemetry_probe import HardwareLogicProbe, ProbedNeuronALU
from cpu_step3_registers import NeuronRegister

class ClockGenerator:
    """
    Krystalový oscilátor a generátor fázovaných hodinových impulzů.
    """
    def __init__(self, target_freq_hz: float = 4_000_000.0): # 4 MHz virtuální krystal
        self.target_freq_hz = target_freq_hz
        self.virtual_period_ns = 1_000_000_000.0 / target_freq_hz
        self.total_t_states = 0
        self.wall_start_perf = time.perf_counter()

    def tick(self, t_states_cost: int = 1):
        self.total_t_states += t_states_cost

    def get_telemetry(self):
        wall_elapsed = time.perf_counter() - self.wall_start_perf
        real_freq_hz = self.total_t_states / wall_elapsed if wall_elapsed > 0 else 0.0
        virtual_sim_time_ms = (self.total_t_states * self.virtual_period_ns) / 1_000_000.0
        return {
            "total_t_states": self.total_t_states,
            "wall_elapsed_s": wall_elapsed,
            "real_emulation_freq_khz": real_freq_hz / 1000.0,
            "virtual_clock_freq_mhz": self.target_freq_hz / 1_000_000.0,
            "virtual_time_ms": virtual_sim_time_ms
        }


class TuringNeuronCPU:
    """
    Turingovsky úplný neuronový procesor s oddělenou nebo rozšířenou RAM.
    """
    def __init__(self, probe: HardwareLogicProbe | None = None, clock_freq_hz: float = 4_000_000.0, ram_size: int = 64):
        self.probe = probe or HardwareLogicProbe()
        self.clock = ClockGenerator(target_freq_hz=clock_freq_hz)
        
        # 1. Neuronová ALU
        self.alu = ProbedNeuronALU(self.probe)
        
        # 2. Kalibrace NOR paměťových prvků pro registry
        self.socket = nn.NeuronCore(input_dim=2, learning_rate=1.5, seed=42)
        X_nor = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
        y_nor = np.array([1, 0, 0, 0])
        self.socket.train_fast(X_nor, y_nor, max_cas_sekund=0.5, target_loss=1e-3, batch_epochs=100)
        self.nor_W = self.socket.W.copy()
        self.nor_b = float(self.socket.b)

        # 3. Registry procesoru (4-bitový akumulátor, 8-bitový PC)
        self.ACC = NeuronRegister(num_bits=4, core=self.socket, nor_weights=self.nor_W, nor_bias=self.nor_b)
        self.ACC.write(0)
        self.PC = 0
        
        # Příznakové registry (Status Flags)
        self.flag_Z = 0 # Zero Flag
        self.flag_C = 0 # Carry Flag
        
        # Operační paměť RAM (64 bajtů)
        self.RAM_SIZE = ram_size
        self.RAM = [0] * ram_size
        self.halted = False
        self.instructions_executed = 0

    def load_program(self, program: list[tuple[int, int]]):
        for i, (op, oper) in enumerate(program):
            self.RAM[i * 2] = op
            self.RAM[i * 2 + 1] = oper

    def _update_flags(self, result_4bit: int, carry_out: int = 0):
        self.flag_Z = 1 if (result_4bit & 0xF) == 0 else 0
        self.flag_C = 1 if carry_out else 0

    def step(self, verbose: bool = True):
        if self.halted:
            return False

        self.instructions_executed += 1
        
        # FETCH
        self.clock.tick(t_states_cost=2)
        pc_addr = self.PC
        opcode = self.RAM[self.PC]
        operand = self.RAM[self.PC + 1]
        self.PC = (self.PC + 2) % self.RAM_SIZE

        inst_name = "UNK"
        detail = ""

        if opcode == 0x1: # LOAD [addr]
            self.clock.tick(2)
            inst_name = f"LOAD [0x{operand:X}]"
            val = self.RAM[operand]
            self.ACC.write(val)
            self.flag_Z = 1 if (val & 0xF) == 0 else 0
            # POZOR: LOAD v procesorech (x86, Z80, 6502) NEMĚNÍ CARRY FLAG!
            detail = f"ACC <- RAM[0x{operand:X}] ({val}), Z={self.flag_Z}"

        elif opcode == 0x2: # ADD [addr]
            self.clock.tick(3)
            inst_name = f"ADD  [0x{operand:X}]"
            acc_val = self.ACC.read()
            ram_val = self.RAM[operand]
            
            # 4-bitové sčítání přes neuronovou ALU (kaskáda 4 bitů s přenosem)
            s_bits = []
            c = 0
            for bit_idx in range(4):
                a_b = (acc_val >> bit_idx) & 1
                b_b = (ram_val >> bit_idx) & 1
                # Full adder: (a + b) + c
                s_raw, c_a = self.alu.half_adder_probed(a_b, b_b)
                s_final, c_b = self.alu.half_adder_probed(s_raw, c)
                c = self.alu.eval_probed(f"ALU_Bit{bit_idx}_Cout", "OR", c_a, c_b)
                s_bits.append(s_final)
            
            res_val = (s_bits[0] | (s_bits[1] << 1) | (s_bits[2] << 2) | (s_bits[3] << 3)) & 0xF
            self.ACC.write(res_val)
            self._update_flags(res_val, carry_out=c)
            detail = f"ALU: {acc_val} + {ram_val} = {res_val}, Z={self.flag_Z}, C={self.flag_C}"

        elif opcode == 0x6: # SUB [addr]
            self.clock.tick(4)
            inst_name = f"SUB  [0x{operand:X}]"
            acc_val = self.ACC.read()
            ram_val = self.RAM[operand]
            
            # A - B
            res_val = (acc_val - ram_val) & 0xF
            self.ACC.write(res_val)
            self._update_flags(res_val)
            detail = f"ALU: {acc_val} - {ram_val} = {res_val}, Z={self.flag_Z}"

        elif opcode == 0x3: # STORE [addr]
            self.clock.tick(2)
            inst_name = f"STORE [0x{operand:X}]"
            acc_val = self.ACC.read()
            self.RAM[operand] = acc_val
            detail = f"RAM[0x{operand:X}] <- {acc_val}"

        elif opcode == 0x4: # JMP [addr]
            self.clock.tick(1)
            inst_name = f"JMP  0x{operand:X}"
            self.PC = operand
            detail = f"PC <- 0x{operand:X}"

        elif opcode == 0x5: # JNZ [addr]
            self.clock.tick(2)
            inst_name = f"JNZ  0x{operand:X}"
            if self.flag_Z == 0:
                self.PC = operand
                detail = f"SKOK: Z=0 -> PC <- 0x{operand:X}"
            else:
                detail = f"IGNOROVÁNO: Z=1 (Konec cyklu)"

        elif opcode == 0xF: # HALT
            self.clock.tick(1)
            inst_name = "HALT"
            self.halted = True
            detail = "HALT"

        if verbose:
            acc_disp = self.ACC.read()
            t_stat = self.clock.total_t_states
            print(f"  Takt #{t_stat:03d} | PC: 0x{pc_addr:X} | {inst_name:<15} | ACC: {acc_disp} (0b{bin(acc_disp)[2:].zfill(4)}) | FLAGS:[Z={self.flag_Z}, C={self.flag_C}] | {detail}")

        return not self.halted
