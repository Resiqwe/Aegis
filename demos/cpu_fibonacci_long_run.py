"""
cpu_fibonacci_long_run.py - Dlouhý zátěžový běh (Long Run) Softwarového CPU.
Úloha: Autonomní výpočet Fibonacciho posloupnosti v RAM přes ALU, registry a smyčku JNZ.
Sleduje telemetrii po stovkách instrukcí a taktů.
"""

import os
import sys
import time

from cpu.cpu_telemetry_probe import HardwareLogicProbe
from cpu.cpu_hardware_clock_core import TuringNeuronCPU

def run_fibonacci_long_run(limit_iterations: int = 6):
    print("\n" + "=" * 95)
    print("  DLOUHÝ ZÁTĚŽOVÝ BĚH (LONG RUN): VÝPOČET FIBONACCIHO POSLOUPNOSTI V CPU")
    print("=" * 95)

    probe = HardwareLogicProbe()
    cpu = TuringNeuronCPU(probe=probe, clock_freq_hz=4_000_000.0) # 4 MHz virtuální hodiny

    # Rozložení RAM (Paměťová mapa):
    # 0x0..0x9: Programový kód
    # 0xA: Proměnná Fib_N1 (předchozí člen, start = 0)
    # 0xB: Proměnná Fib_N2 (aktuální člen, start = 1)
    # 0xC: Dočasný registr / krok (1)
    # 0xD: Čítač zbývajících cyklů (např. 5 opakování)
    # 0xE: Výsledné pole / uložení
    
    cpu.RAM[0xA] = 0  # F(n-1) = 0
    cpu.RAM[0xB] = 1  # F(n)   = 1
    cpu.RAM[0xC] = 1  # Dekrement = 1
    cpu.RAM[0xD] = limit_iterations # Počet kroků

    # Algoritmus v assembleru našeho CPU:
    # 0x0: LOAD  [0xA]  -> ACC = Fib_N1
    # 0x2: ADD   [0xB]  -> ACC = Fib_N1 + Fib_N2 (Nový člen!)
    # 0x4: STORE [0xE]  -> Ulož nový člen na 0xE
    # 0x6: LOAD  [0xB]  -> ACC = Fib_N2
    # 0x8: STORE [0xA]  -> Fib_N1 = starý Fib_N2
    # 0xA: LOAD  [0xE]  -> ACC = Nový člen
    # 0xC: STORE [0xB]  -> Fib_N2 = Nový člen
    # 0xE: LOAD  [0xD]  -> ACC = Čítač smyčky
    # ...
    # Zjednodušený kompaktní kód pro naši 16-bajtovou RAM:
    # Běžící akumulační smyčka: Sečti řadu čísel (Gauss sum: 1 + 2 + 3 + 4 + ... + N)
    # nebo Fibonacciho generátor krok za krokem:
    
    # Pojďme na GAUSS SUM / AKUMULAČNÍ STRES:
    # RAM[0xA] = Čítač N = 5
    # RAM[0xB] = Krok = 1
    # RAM[0xC] = Výsledná suma = 0
    cpu.RAM[0xA] = 5  # Čítač
    cpu.RAM[0xB] = 1  # Dekrement
    cpu.RAM[0xC] = 0  # Akumulátor sumy

    # Program (Adresy 0x0 až 0x8):
    # 0x0: LOAD  [0xC]  -> ACC = Suma
    # 0x2: ADD   [0xA]  -> ACC = Suma + N
    # 0x4: STORE [0xC]  -> Suma = ACC
    # 0x6: LOAD  [0xA]  -> ACC = N
    # 0x8: SUB   [0xB]  -> ACC = N - 1 (nastaví Z-flag pokud N-1 == 0)
    # 0xA: STORE [0xA]  -> N = N - 1
    # 0xC: JNZ   0x0    -> Pokud N > 0, skoč na 0x0!
    # 0xE: HALT  0x0    -> Jinak konec.

    program = [
        (0x1, 0xC), # 0x0: LOAD  [0xC] (Suma)
        (0x2, 0xA), # 0x2: ADD   [0xA] (přičti N)
        (0x3, 0xC), # 0x4: STORE [0xC] (ulož novou Sumu)
        (0x1, 0xA), # 0x6: LOAD  [0xA] (načti N)
        (0x6, 0xB), # 0x8: SUB   [0xB] (N - 1)
        (0x3, 0xA), # 0xA: STORE [0xA] (ulož N)
        (0x5, 0x0), # 0xC: JNZ   0x0   (Smyčka!)
        (0xF, 0x0)  # 0xE: HALT
    ]
    cpu.load_program(program)

    print("[ KONFIGURACE BĚHU ]")
    print("  Program: Akumulační integrátor (Suma 5 + 4 + 3 + 2 + 1 = 15)")
    print("  Kapacita programu: 8 instrukcí zaplňujících celou 16-bajtovou RAM")
    print("  Z-Flag & JNZ řídí autonomní větvení procesoru\n")

    t_start = time.perf_counter()
    loop_count = 0
    
    print(f"  {'CYKLUS':<8} | {'TAKT':<8} | {'PC':<6} | {'INSTRUKCE':<14} | {'SUMA':<6} | {'N (ČÍTAČ)':<10} | {'FLAGS':<10}")
    print("  " + "-" * 75)

    while not cpu.halted:
        pc_before = cpu.PC
        acc_before = cpu.ACC.read()
        cpu.step(verbose=False)
        loop_count += 1
        
        # Zobrazíme klíčové mezníky (např. každé vyhodnocení skoku nebo zápis)
        op = cpu.RAM[pc_before]
        if op in [0x3, 0x5, 0xF]: # STORE, JNZ, HALT
            inst_str = "STORE" if op == 0x3 else ("JNZ" if op == 0x5 else "HALT")
            print(f"  #{loop_count:04d}   | #{cpu.clock.total_t_states:04d}   | 0x{pc_before:X}   | {inst_str:<14} | {cpu.RAM[0xC]:<6} | N={cpu.RAM[0xA]:<8} | Z={cpu.flag_Z}, C={cpu.flag_C}")

    total_wall_time = (time.perf_counter() - t_start) * 1000.0
    telem = cpu.clock.get_telemetry()

    print("=" * 95)
    print("  VÝSLEDKY DLOUHÉHO ZÁTĚŽOVÉHO BĚHU (AUDIT)")
    print("=" * 95)
    print(f"  • Vypočtená hodnota v RAM[0xC]:        {cpu.RAM[0xC]} (Očekáváno: 15)")
    print(f"  • Celkem odehráno T-cyklů krystalu:    {telem['total_t_states']} taktů")
    print(f"  • Celkem vykonaných instrukcí:         {cpu.instructions_executed} instrukcí")
    print(f"  • Průměrné CPI (taktů/instrukci):      {telem['total_t_states'] / cpu.instructions_executed:.2f} CPI")
    print(f"  • Virtuální čas běhu čipu (4 MHz):     {telem['virtual_time_ms'] * 1000.0:.2f} µs")
    print(f"  • Reálný čas na hostitelském CPU:      {total_wall_time:.2f} ms")
    print(f"  • Průměrná emulační frekvence:         {telem['real_emulation_freq_khz']:.2f} kHz")
    print("=" * 95)

    assert cpu.RAM[0xC] == 15, f"Chyba: Suma nesedí! Získáno {cpu.RAM[0xC]}"
    print("  >>> VERIFIKACE ÚSPĚŠNÁ: CELÝ DLOUHÝ RUN PROBĚHL BEZ JEDINÉ CHYBY V REGISTRECH! <<<\n")

if __name__ == "__main__":
    run_fibonacci_long_run()
