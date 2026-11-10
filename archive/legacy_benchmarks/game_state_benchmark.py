"""
game_state_benchmark.py - První reálná práce pro náš Tenzorový Křemík (Tensor Blob) na RTX 5070 Ti:
Simulace herního světa: 1 000 000 jednotek (agentů) naráz.

Každý herní agent má 4 vstupní binární senzory:
  S0: Vidí nepřítele? (0 = ne, 1 = ano)
  S1: Má málo zdraví (pod 20 %)? (0 = plné/ok, 1 = kritické)
  S2: Má v inventáři lékárničku? (0 = ne, 1 = ano)
  S3: Má nabitou zbraň? (0 = prázdná, 1 = nabitá)

Neuronový Bloček (Decision Core) vyhodnotí stav a vrátí 3 akční signály (výstupy):
  A0: ÚTOK (1 = zahájit střelbu)       -> Podmínka: Vidí nepřítele A má nabito A nemá kritické HP
  A1: ÚTĚK (1 = zahájit sprint vzad)  -> Podmínka: Vidí nepřítele A (nemá náboje NEBO má kritické HP)
  A2: LÉČIT (1 = aplikovat medkit)     -> Podmínka: Má kritické HP A má lékárničku

Proč je to ideální benchmark:
  - 16 různých kombinací herních situací.
  - Testuje složitou nelineární rozhodovací logiku (AND/OR/NOT podmínky).
  - Simulujeme 1 000 000 jednotek najednou v reálném čase přímo v RTX 5070 Ti.
"""

import time
import itertools
import numpy as np
import torch
from brick import Brick

# 1. Definice herních pravidel (pravdivostní tabulka pro herního bota)
def game_ai_rules(s0_enemy, s1_low_hp, s2_has_medkit, s3_has_ammo):
    # A0: ÚTOK = nepřítel & nabito & není kritické HP
    attack = int(s0_enemy and s3_has_ammo and not s1_low_hp)
    # A1: ÚTĚK = nepřítel & (nemá náboje OR má kritické HP)
    flee = int(s0_enemy and (not s3_has_ammo or s1_low_hp))
    # A2: LÉČIT = má kritické HP & má medkit
    heal = int(s1_low_hp and s2_has_medkit)
    return (attack, flee, heal)

def train_game_ai_blob():
    print("[*] Trénuji herní rozhodovací jádro (Herní AI Blob)...")
    # 4 vstupy -> 3 výstupy. Hledáme kompaktní architekturu.
    brick = Brick.learn(
        name="GAME_AI_CORE",
        n_in=4,
        n_out=3,
        fn=game_ai_rules,
        hidden=[8], # 8 skrytých neuronů na složitou logiku
        max_epochs=5000,
        lr=0.08,
        attempts=20
    )
    st = brick.stats()
    print(f"    -> Herní jádro naučeno! Neurony: {st['neurons']}, Vah celkem: {st['params']} (cca {st['params']*4} bajtů)")
    return brick

def run_benchmark():
    print("=" * 85)
    print(" HERNÍ BENCHMARK: MASIVNÍ ROZHODOVÁNÍ 1 000 000 JEDNOTEK (BOTŮ) V REÁLNÉM ČASE")
    print("=" * 85)

    brick = train_game_ai_blob()

    # Příprava 1 000 000 herních situací
    N = 1_000_000
    print(f"\n[*] Generuji herní svět s {N:,} agenty...")
    np.random.seed(42)
    # Senzory náhodně pro 1M agentů
    sensors_cpu = np.random.randint(0, 2, (N, 4), dtype=np.int8)

    # --------------------------------------------------------------------------
    # 1. Klasický CPU kód (jak by to dělal běžný herní engine v Pythonu po jednom)
    # --------------------------------------------------------------------------
    print("[*] [1/2] Simuluji rozhodování na CPU (sekvenční logika herních botů)...")
    n_sample = 100_000
    t0 = time.perf_counter()
    cpu_decisions = []
    for i in range(n_sample):
        s0, s1, s2, s3 = sensors_cpu[i]
        cpu_decisions.append(game_ai_rules(s0, s1, s2, s3))
    dt_cpu = (time.perf_counter() - t0) * (N / n_sample)
    fps_cpu = 1.0 / dt_cpu if dt_cpu > 0 else 0

    print(f"    -> Čas CPU pro 1 000 000 agentů:  {dt_cpu * 1000:.1f} ms ({fps_cpu:.1f} FPS pro 1M agentů)")

    # --------------------------------------------------------------------------
    # 2. Tenzorový Blob na RTX 5070 Ti (CUDA)
    # --------------------------------------------------------------------------
    print("\n[*] [2/2] Posílám herní svět do RTX 5070 Ti (Tensor AI Blob)...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Váhy blobu na GPU
    W1 = torch.tensor(brick.params[0][0], dtype=torch.float32, device=device)
    b1 = torch.tensor(brick.params[0][1], dtype=torch.float32, device=device)
    W2 = torch.tensor(brick.params[1][0], dtype=torch.float32, device=device)
    b2 = torch.tensor(brick.params[1][1], dtype=torch.float32, device=device)

    # 1M agentů nahráno do VRAM
    sensors_gpu = torch.from_numpy(sensors_cpu.astype(np.float32)).to(device)

    # Zahřátí GPU
    _ = torch.sigmoid(torch.tanh(sensors_gpu[:1000] @ W1 + b1) @ W2 + b2)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    # ROZHODNUTÍ CELÉHO HERNÍHO SVĚTA V JEDINÉM ZÁBLESKU:
    h = torch.tanh(sensors_gpu @ W1 + b1)
    actions_gpu = (torch.sigmoid(h @ W2 + b2) >= 0.5).to(torch.int8)
    torch.cuda.synchronize()
    dt_gpu = time.perf_counter() - t0
    fps_gpu = 1.0 / dt_gpu if dt_gpu > 0 else 0

    print(f"    -> Čas RTX 5070 Ti pro 1 000 000: {dt_gpu * 1000:.2f} ms")
    print(f"    -> Reálný herní framerate AI:    {fps_gpu:,.0f} FPS (snímků za sekundu pro 1M botů!)")

    # --------------------------------------------------------------------------
    # Kontrola správnosti
    # --------------------------------------------------------------------------
    print("\n[*] Kontroluji správnost taktických rozhodnutí botů...")
    sample_actions = actions_gpu[:100_000].cpu().numpy()
    correct = 0
    for i in range(100_000):
        s0, s1, s2, s3 = sensors_cpu[i]
        expected = game_ai_rules(s0, s1, s2, s3)
        actual = tuple(sample_actions[i])
        if expected == actual:
            correct += 1

    print(f"    -> Přesnost rozhodování: {correct / 100_000 * 100:.2f}% ({correct}/100 000)")

    print("\n" + "=" * 85)
    print(" SHRNUTÍ VÝSLEDKŮ:")
    print("=" * 85)
    print(f"  CPU klasický engine:   1 000 000 botů zabere {dt_cpu * 1000:.1f} ms  (hra by běžela na {fps_cpu:.1f} FPS - sekala by se)")
    print(f"  RTX 5070 Ti AI Blob:   1 000 000 botů zabere {dt_gpu * 1000:.2f} ms  (hra běží na {fps_gpu:,.0f} FPS - naprosto plynule!)")
    print(f"  Zrychlení:             RTX je {dt_cpu / dt_gpu:.1f}x RYCHLEJŠÍ!")
    print(f"  Paměť herního mozku:   Pouhých {W1.numel()*4 + b1.numel()*4 + W2.numel()*4 + b2.numel()*4} BAJTŮ!")
    print("=" * 85)

if __name__ == "__main__":
    run_benchmark()
