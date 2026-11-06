"""
cpu_stress_adc_multibyte.py - Náročná vícebajtová (8-bit) aritmetika na 4-bitovém procesoru.
Demonstruje:
  1. Přetečení 4-bitového registru: 8 + 8 = 16 (ACC=0, Carry=1)
  2. Implementaci instrukce ADC (Add with Carry) v neuronové ALU
  3. Plnohodnotné 8-bitové sčítání na 4-bitovém čipu:
     Např. 45 + 85 = 130  (v hexu: 0x2D + 0x55 = 0x82)
     Rozpadnuto na spodní (Low Nibble) a horní (High Nibble) 4 bity s přenosem!
"""

import os
import sys
import time

sys.path.append(os.path.dirname(__file__))
from cpu_telemetry_probe import HardwareLogicProbe
from cpu_hardware_clock_core import TuringNeuronCPU

def run_overflow_and_adc_stress():
    print("\n" + "=" * 95)
    print("  NÁROČNÁ OPERACE: VÍCEBAJTOVÁ ARITMETIKA (8-BIT MATEMATIKA NA 4-BITOVÉM KŘEMÍKU)")
    print("=" * 95)

    probe = HardwareLogicProbe()
    cpu = TuringNeuronCPU(probe=probe, clock_freq_hz=4_000_000.0, ram_size=64)

    # --------------------------------------------------------------------------
    # ČÁST 1: DEMONSTRACE PŘETEČENÍ (8 + 8)
    # --------------------------------------------------------------------------
    print("\n[ TEST 1: PŘETEČENÍ PŘI 8 + 8 NA 4 BITECH ]")
    print("  4-bitový registr udrží max 15. Zkoušíme sečíst 8 + 8...")

    ADDR_A = 0x20
    ADDR_B = 0x21
    ADDR_RES = 0x22

    cpu.RAM[ADDR_A] = 8
    cpu.RAM[ADDR_B] = 8

    # Program: LOAD [A], ADD [B], STORE [RES], HALT
    cpu.load_program([
        (0x1, ADDR_A),   # 0x0: LOAD 8
        (0x2, ADDR_B),   # 0x2: ADD 8
        (0x3, ADDR_RES), # 0x4: STORE
        (0xF, 0x0)       # 0x6: HALT
    ])

    while cpu.step(verbose=False):
        pass

    raw_acc = cpu.RAM[ADDR_RES]
    carry = cpu.flag_C
    print(f"  -> Výsledek v ACC:         {raw_acc} (0b{bin(raw_acc)[2:].zfill(4)})")
    print(f"  -> Stav Carry Flagu:       C = {carry} (PŘÍZNAK PŘETEČENÍ SEPNUT!)")
    print(f"  -> Skutečná hodnota:       (Carry << 4) + ACC = ({carry} * 16) + {raw_acc} = {(carry * 16) + raw_acc}")
    print("  Závěr: Registr přetekl, 5. bit visí na příznaku Carry!\n")

    # --------------------------------------------------------------------------
    # ČÁST 2: SKUTEČNÁ 8-BITOVÁ MAKAČKA (45 + 85 = 130)
    # --------------------------------------------------------------------------
    print("=" * 95)
    print("[ TEST 2: SKUTEČNÁ 8-BITOVÁ MATEMATIKA PŘES DVOU-KROKOVOU KASKÁDU ]")
    print("  Sčítáme velká čísla: 45 + 85 = 130")
    print("  Rozpad na 4-bitové nibbly (šestnáctkově):")
    print("    45 = 0x2D -> High Nibble: 0x2 (2), Low Nibble: 0xD (13)")
    print("    85 = 0x55 -> High Nibble: 0x5 (5), Low Nibble: 0x5 (5)")
    print("  Očekávaný součet 130 = 0x82:")
    print("    Low:  13 + 5 = 18 -> 18 mod 16 = 2, Carry = 1")
    print("    High: 2 + 5 + Carry(1) = 8\n")

    # Paměťová mapa pro 8-bitovou kalkulaci:
    # 0x20: A_low  = 0xD (13)
    # 0x21: A_high = 0x2 (2)
    # 0x22: B_low  = 0x5 (5)
    # 0x23: B_high = 0x5 (5)
    # 0x24: Res_low
    # 0x25: Res_high
    # 0x26: Temp_carry
    # 0x27: Konstanta 1
    cpu = TuringNeuronCPU(probe=probe, clock_freq_hz=4_000_000.0, ram_size=64)
    cpu.RAM[0x20] = 0xD
    cpu.RAM[0x21] = 0x2
    cpu.RAM[0x22] = 0x5
    cpu.RAM[0x23] = 0x5
    cpu.RAM[0x27] = 1

    # Program emulující vícebajtovou instrukci ADC (Add with Carry):
    # 1. Sečti spodní bity:
    #    LOAD [0x20] (A_low)
    #    ADD  [0x22] (B_low) -> generuje výsledek a Carry Flag!
    #    STORE [0x24] (Res_low)
    # 2. Ulož stav Carry (pokud C=1, skoč a přičti 1 k horním bitům):
    #    JNZ_CARRY větev...
    
    # Pro přehlednost zapojíme rozšířenou instrukční podporu přímo v CPU:
    # Přidáme instrukci ADC (0x7 [addr]): ACC = ACC + RAM[addr] + Carry
    # A spustíme program:
    program_adc = [
        (0x1, 0x20), # 0x00: LOAD A_low
        (0x2, 0x22), # 0x02: ADD  B_low  (spodní 4 bity, přenos do C)
        (0x3, 0x24), # 0x04: STORE Res_low
        (0x1, 0x21), # 0x06: LOAD A_high
        (0x7, 0x23), # 0x08: ADC  B_high (horní 4 bity + starý Carry!)
        (0x3, 0x25), # 0x0A: STORE Res_high
        (0xF, 0x00)  # 0x0C: HALT
    ]

    # Implementujeme instrukci 0x7 (ADC) do CPU jádra
    original_step = cpu.step
    def step_with_adc(verbose=False):
        if cpu.halted:
            return False
        opcode = cpu.RAM[cpu.PC]
        operand = cpu.RAM[cpu.PC + 1]
        
        if opcode == 0x7: # ADC [addr]
            cpu.instructions_executed += 1
            cpu.clock.tick(t_states_cost=4) # ADC stojí 4 takty (složitější hradlová cesta)
            pc_addr = cpu.PC
            cpu.PC = (cpu.PC + 2) % cpu.RAM_SIZE
            
            acc_val = cpu.ACC.read()
            ram_val = cpu.RAM[operand]
            old_carry = cpu.flag_C
            
            # Sčítáme ACC + RAM + old_carry přes Full-Adder kaskádu
            s_bits = []
            c = old_carry
            for bit_idx in range(4):
                a_b = (acc_val >> bit_idx) & 1
                b_b = (ram_val >> bit_idx) & 1
                s_raw, c_a = cpu.alu.half_adder_probed(a_b, b_b)
                s_final, c_b = cpu.alu.half_adder_probed(s_raw, c)
                c = cpu.alu.eval_probed(f"ADC_Bit{bit_idx}_Cout", "OR", c_a, c_b)
                s_bits.append(s_final)
                
            res_val = (s_bits[0] | (s_bits[1] << 1) | (s_bits[2] << 2) | (s_bits[3] << 3)) & 0xF
            cpu.ACC.write(res_val)
            cpu._update_flags(res_val, carry_out=c)
            if verbose:
                print(f"  ADC: {acc_val} + {ram_val} + Carry({old_carry}) = {res_val} -> ACC, New Carry={c}")
            return True
        else:
            return original_step(verbose=verbose)

    cpu.step = step_with_adc
    cpu.load_program(program_adc)

    t0 = time.perf_counter()
    while cpu.step(verbose=True):
        pass
    elapsed = (time.perf_counter() - t0) * 1000

    low_res = cpu.RAM[0x24]
    high_res = cpu.RAM[0x25]
    final_8bit = (high_res << 4) | low_res

    print("\n" + "=" * 95)
    print("  VÝSLEDEK 8-BITOVÉ OPERACE:")
    print("=" * 95)
    print(f"  • Spodní nibble (Low 4-bit) v RAM[0x24]:   0x{low_res:X} ({low_res})")
    print(f"  • Horní nibble (High 4-bit) v RAM[0x25]:  0x{high_res:X} ({high_res})")
    print(f"  • Složené 8-bitové číslo:                 0x{final_8bit:02X} = {final_8bit}")
    print(f"  • Očekávaný součet (45 + 85):             130")
    print(f"  • Čas provedení celé kaskády:             {elapsed:.2f} ms")
    print("=" * 95)

    assert final_8bit == 130, f"Chyba: 8-bit výsledek nesedí! Získáno {final_8bit}"
    print("  >>> DOKÁZÁNO: 4-BITOVÝ NEURONOVÝ ČIP ÚSPĚŠNĚ SPOČETL VELKOU 8-BITOVOU ÚLOHU! <<<\n")

if __name__ == "__main__":
    run_overflow_and_adc_stress()
