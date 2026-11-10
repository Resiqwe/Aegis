"""
exp_adders.py - Experiment: Tabulka vs. Naučená síť (Kostička) vs. Skládaná síť

Porovnává:
1. 1-bitová sčítačka (Full Adder: 3 vstupy -> 2 výstupy)
2. 2-bitová sčítačka (4 vstupy -> 3 výstupy)
3. 3-bitová sčítačka (6 vstupů -> 4 výstupy)
4. Skládaná 2-bit a 4-bit sčítačka (řetězení 1-bitových Full Adder kostiček)
"""

import time
import itertools
from brick import Brick

def make_adder_spec(bits: int):
    n_in = 2 * bits
    n_out = bits + 1
    def fn(*inputs):
        a_bits = inputs[:bits]
        b_bits = inputs[bits:]
        a = sum(bit << i for i, bit in enumerate(a_bits))
        b = sum(bit << i for i, bit in enumerate(b_bits))
        total = a + b
        return tuple((total >> i) & 1 for i in range(n_out))
    return n_in, n_out, fn

def run_experiment():
    print("=" * 75)
    print(" EXPERIMENT: TABULKA vs. NAUČENÁ KOSTIČKA (SÍŤ)")
    print("=" * 75)
    
    # 1. Monolitické kostičky (naučené naráz)
    monolithic_configs = [
        ("ADD_1bit", 1, [6], 5000),
        ("ADD_2bit", 2, [12], 5000),
        ("ADD_3bit", 3, [24], 8000),
    ]
    
    results = []
    
    for name, bits, hidden, max_ep in monolithic_configs:
        n_in, n_out, fn = make_adder_spec(bits)
        table_rows = 2 ** n_in
        table_values = table_rows * (n_in + n_out)
        
        print(f"\n[+] Trénuji monolitickou kostičku: {name} ({bits}-bit)...")
        t0 = time.perf_counter()
        try:
            b = Brick.learn(name, n_in, n_out, fn, hidden=hidden, max_epochs=max_ep, lr=0.08, attempts=15)
            dt = time.perf_counter() - t0
            st = b.stats()
            results.append({
                "typ": f"Monolit {bits}-bit",
                "vstupy": n_in,
                "výstupy": n_out,
                "tabulka_hodnot": table_values,
                "neurony": st["neurons"],
                "parametry_vahy": st["params"],
                "epochy": st["epochs"],
                "cas_s": dt,
                "overeno": "ANO (100%)"
            })
            print(f"    -> OK! Naučeno za {dt:.2f}s ({st['epochs']} epoch, {st['params']} parametrů/vah)")
        except RuntimeError as e:
            print(f"    -> Selhalo: {e}")

    # 2. Skládaná sčítačka (Lego z 1-bitových Full Adderů)
    print("\n[+] Skládám 2-bit a 4-bit sčítačku z 1-bitových Full Adder kostiček...")
    # Naučíme základní Full Adder
    fa = Brick.learn("FA_BASE", 3, 2, lambda a, b, cin: ((a + b + cin) & 1, (a + b + cin) >> 1), hidden=[6])
    
    # Skládaná funkce
    def ripple_add(a_val, b_val, bits):
        c = 0
        res = 0
        for i in range(bits):
            a_bit = (a_val >> i) & 1
            b_bit = (b_val >> i) & 1
            s_bit, c = fa(a_bit, b_bit, c)
            res |= (s_bit << i)
        res |= (c << bits)
        return res
    
    # Ověření složené 2-bit sčítačky
    ok_2bit = all(ripple_add(a, b, 2) == a + b for a in range(4) for b in range(4))
    fa_st = fa.stats()
    results.append({
        "typ": "Složená 2-bit (2x FA)",
        "vstupy": 4,
        "výstupy": 3,
        "tabulka_hodnot": 16 * 7,
        "neurony": fa_st["neurons"] * 2,
        "parametry_vahy": fa_st["params"] * 2,
        "epochy": f"{fa_st['epochs']} (jen 1x)",
        "cas_s": fa_st["seconds"],
        "overeno": f"ANO ({ok_2bit})"
    })
    
    # Ověření složené 4-bit sčítačky
    ok_4bit = all(ripple_add(a, b, 4) == a + b for a in range(16) for b in range(16))
    results.append({
        "typ": "Složená 4-bit (4x FA)",
        "vstupy": 8,
        "výstupy": 5,
        "tabulka_hodnot": 256 * 13,
        "neurony": fa_st["neurons"] * 4,
        "parametry_vahy": fa_st["params"] * 4,
        "epochy": f"{fa_st['epochs']} (jen 1x)",
        "cas_s": fa_st["seconds"],
        "overeno": f"ANO ({ok_4bit})"
    })

    print("\n" + "=" * 95)
    print(f"{'Typ architektury':<22} | {'Vst/Výst':<8} | {'Tabulka':<10} | {'Neurony':<8} | {'Parametry':<10} | {'Čas/Epochy'}")
    print("-" * 95)
    for r in results:
        vyst = f"{r['vstupy']}->{r['výstupy']}"
        tab = f"{r['tabulka_hodnot']} b"
        ce = f"{r['cas_s']:.2f}s ({r['epochy']})"
        print(f"{r['typ']:<22} | {vyst:<8} | {tab:<10} | {r['neurony']:<8} | {r['parametry_vahy']:<10} | {ce}")
    print("=" * 95)

if __name__ == "__main__":
    run_experiment()
