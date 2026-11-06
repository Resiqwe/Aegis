"""
cpu_gauss_sum_long_run.py - Dlouhý zátěžový běh (Long Run) Softwarového CPU.
Úloha: Autonomní akumulační integrátor (Gauss sum: 3 + 2 + 1 = 6 nebo 5 + 4 + 3 + 2 + 1 = 15).
Využívá rozšířenou RAM s odděleným kódem a daty (Harvard/von Neumann hygiena).
"""

import os
import sys
import time

from cpu.cpu_telemetry_probe import HardwareLogicProbe
from cpu.cpu_hardware_clock_core import TuringNeuronCPU

def run_gauss_long_run(n_start: int = 3):
    print("\n" + "=" * 95)
    print(f"  DLOUHÝ ZÁTĚŽOVÝ BĚH (LONG RUN): AKUMULAČNÍ SMYČKA V NEURONOVÉM CPU (N={n_start})")
    print("=" * 95)

    probe = HardwareLogicProbe()
    cpu = TuringNeuronCPU(probe=probe, clock_freq_hz=4_000_000.0, ram_size=64)

    # Paměťová mapa:
    # Kód je na adresách: 0x00 .. 0x0F
    # Data jsou bezpečně v datovém segmentu od adresy 0x20 výše!
    # 0x20: Čítač N
    # 0x21: Krok (1)
    # 0x22: Akumulátor Sumy (start: 0)
    ADDR_N    = 0x20
    ADDR_KROK = 0x21
    ADDR_SUMA = 0x22

    cpu.RAM[ADDR_N]    = n_start
    cpu.RAM[ADDR_KROK] = 1
    cpu.RAM[ADDR_SUMA] = 0

    # Program:
    # 0x00: LOAD  [ADDR_SUMA] -> ACC = Suma
    # 0x02: ADD   [ADDR_N]    -> ACC = Suma + N
    # 0x04: STORE [ADDR_SUMA] -> Suma = ACC
    # 0x06: LOAD  [ADDR_N]    -> ACC = N
    # 0x08: SUB   [ADDR_KROK] -> ACC = N - 1 (nastaví Z-flag pokud N-1 == 0)
    # 0x0A: STORE [ADDR_N]    -> N = N - 1
    # 0x0C: JNZ   0x00        -> Pokud N > 0, skoč zpět na 0x00!
    # 0x0E: HALT  0x00        -> Konec
    program = [
        (0x1, ADDR_SUMA), # 0x00
        (0x2, ADDR_N),    # 0x02
        (0x3, ADDR_SUMA), # 0x04
        (0x1, ADDR_N),    # 0x06
        (0x6, ADDR_KROK), # 0x08
        (0x3, ADDR_N),    # 0x0A
        (0x5, 0x00),      # 0x0C: JNZ 0x00
        (0xF, 0x00)       # 0x0E: HALT
    ]
    cpu.load_program(program)

    expected_sum = sum(range(1, n_start + 1))
    print(f"[ KONFIGURACE BĚHU ]")
    print(f"  Startovní N = {n_start} | Očekávaná suma = {expected_sum}")
    print(f"  Kódový segment: 0x00 - 0x0E | Datový segment: 0x20 - 0x22\n")

    t_start = time.perf_counter()
    loop_count = 0

    print(f"  {'CYKLUS':<8} | {'TAKT':<8} | {'PC':<6} | {'INSTRUKCE':<16} | {'SUMA':<6} | {'N':<6} | {'FLAGS':<10}")
    print("  " + "-" * 75)

    while not cpu.halted:
        pc_before = cpu.PC
        cpu.step(verbose=False)
        loop_count += 1
        
        op = cpu.RAM[pc_before]
        if op in [0x3, 0x5, 0xF]:
            inst_str = "STORE" if op == 0x3 else ("JNZ" if op == 0x5 else "HALT")
            print(f"  #{loop_count:04d}   | #{cpu.clock.total_t_states:04d}   | 0x{pc_before:02X}   | {inst_str:<16} | {cpu.RAM[ADDR_SUMA]:<6} | {cpu.RAM[ADDR_N]:<6} | Z={cpu.flag_Z}, C={cpu.flag_C}")

    total_wall_time = (time.perf_counter() - t_start) * 1000.0
    telem = cpu.clock.get_telemetry()

    print("\n" + "=" * 95)
    print("  TELEMETRIE DLOUHÉHO BĚHU (AUDIT)")
    print("=" * 95)
    print(f"  • Vypočtená hodnota v RAM[0x22]:       {cpu.RAM[ADDR_SUMA]} (Očekáváno: {expected_sum})")
    print(f"  • Celkem odehráno T-cyklů krystalu:    {telem['total_t_states']} taktů")
    print(f"  • Celkem vykonaných instrukcí:         {cpu.instructions_executed} instrukcí")
    print(f"  • Průměrné CPI (taktů/instrukci):      {telem['total_t_states'] / cpu.instructions_executed:.2f} CPI")
    print(f"  • Virtuální čas běhu čipu (4 MHz):     {telem['virtual_time_ms'] * 1000.0:.2f} µs")
    print(f"  • Reálný čas na hostitelském CPU:      {total_wall_time:.2f} ms")
    print(f"  • Průměrná emulační frekvence:         {telem['real_emulation_freq_khz']:.2f} kHz")
    print("=" * 95)

    assert cpu.RAM[ADDR_SUMA] == expected_sum, f"Chyba: Suma nesedí! Získáno {cpu.RAM[ADDR_SUMA]}"
    print("  >>> VERIFIKACE ÚSPĚŠNÁ: DLOUHÝ RUN PROBĚHL DETERMINISTICKY A BEZ CHYBY! <<<\n")
    return True

if __name__ == "__main__":
    run_gauss_long_run(n_start=5)
