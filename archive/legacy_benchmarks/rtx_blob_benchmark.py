"""
rtx_blob_benchmark.py - Demonstrace "Tensor Silicon Blobu" na NVIDIA RTX 5070 Ti.

Vezme vytuněnou architekturu plné sčítačky (FULL_ADDER: 3 vstupy -> 3 skryté -> 2 výstupy)
z `tune_core.py` a napálí ji do CUDA jader tvé RTX 5070 Ti.

Porovná:
  1. Sekvenční běh (jedno po druhém jak na CPU)
  2. Masivní paralelní bombardování (dávka 10 MILIONŮ sčítacích operací naráz v jednom taktu GPU)
"""

import time
import torch
from tune_core import tune_all

def main():
    print("=" * 80)
    print(" TENZOROVÝ KŘEMÍK (TENSOR BLOB) NA NVIDIA RTX 5070 Ti")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Detekováno zařízení: {torch.cuda.get_device_name(0)}")
    print(f"[*] VRAM celkem:         {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n")

    # 1. Získáme ověřené váhy z našeho CPU ladění
    print("[*] Získávám vytuněné váhy z tune_core...")
    chips = tune_all()
    fa = chips["FULL_ADDER"]
    
    # Převedeme váhy na PyTorch tenzory přímo v paměti GPU (CUDA)
    # Layer 0: W1 (3x3), b1 (3)
    # Layer 1: W2 (3x2), b2 (2)
    W1 = torch.tensor(fa.params[0][0], dtype=torch.float32, device=device)
    b1 = torch.tensor(fa.params[0][1], dtype=torch.float32, device=device)
    W2 = torch.tensor(fa.params[1][0], dtype=torch.float32, device=device)
    b2 = torch.tensor(fa.params[1][1], dtype=torch.float32, device=device)

    print("\n" + "=" * 80)
    print(" EXTRÉMNÍ ZÁTĚŽOVÝ TEST: 10 000 000 (DESET MILIONŮ) OPERACÍ NARÁZ V GPU")
    print("=" * 80)

    N = 10_000_000 # 10 milionů operací
    print(f"[*] Generuji náhodný proud dat ({N:,} trojic bitů A, B, Carry-in)...")
    
    # 10 milionů trojic náhodných bitů 0/1 přímo ve VRAM
    inputs = torch.randint(0, 2, (N, 3), dtype=torch.float32, device=device)

    # Zahřátí GPU (Warmup)
    _ = torch.sigmoid(torch.tanh(inputs[:1000] @ W1 + b1) @ W2 + b2)
    torch.cuda.synchronize()

    print("[*] GPU BOMBARDUJ: Průlet 10 milionů signálů skrze tenzorový křemík...")
    t0 = time.perf_counter()

    # TENTO JEDINÝ ŘÁDEK JE CELÉ „ČIPOVÉ JÁDRO":
    # Signál prolétne skrytou vrstvou a rovnou do výstupu
    h = torch.tanh(inputs @ W1 + b1)
    out_probs = torch.sigmoid(h @ W2 + b2)
    out_bits = (out_probs >= 0.5).to(torch.int8)

    torch.cuda.synchronize() # Čekáme, až GPU dokončí celý výpočet
    dt = time.perf_counter() - t0

    ops_per_sec = N / dt
    vram_used_mb = torch.cuda.memory_allocated() / (1024**2)

    print("\n" + "-" * 80)
    print(" VÝSLEDKY MĚŘENÍ:")
    print("-" * 80)
    print(f"  Čas pro 10 000 000 operací:    {dt * 1000:.2f} milisekund")
    print(f"  Rychlost (propustnost):        {ops_per_sec:,.0f} operací za sekundu ({ops_per_sec / 1e6:.1f} Mops/s)")
    print(f"  Spotřeba VRAM pro 10M prvků:   {vram_used_mb:.1f} MB (z tvých 16 GB!)")
    print(f"  Velikost samotného 'čipu':     {sum(p.numel() * 4 for p in [W1, b1, W2, b2])} BAJTŮ (čistý cucek vah)")
    print("-" * 80)

    # 100% kontrola správnosti na náhodném vzorku 100 000 výsledků
    print("\n[*] Ověřuji matematickou přesnost (100k vzorků)...")
    sample_in = inputs[:100_000].cpu().numpy().astype(int)
    sample_out = out_bits[:100_000].cpu().numpy()
    
    correct = 0
    for (a, b, c), (s, cout) in zip(sample_in, sample_out):
        expected_s = (a + b + c) & 1
        expected_cout = (a + b + c) >> 1
        if s == expected_s and cout == expected_cout:
            correct += 1

    print(f"  Úspěšnost verifikace: {correct / 100_000 * 100:.2f}% (Přesně: {correct}/100 000)")
    print("=" * 80)

if __name__ == "__main__":
    main()
