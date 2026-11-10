"""
arena_sim.py - Vizuální terminálová aréna řízená Tenzorovým Křemíkem (Tensor Blob) na RTX 5070 Ti!

Herní princip:
  - V aréně běhá 40 jednotek (boti) označených písmeny.
  - V každém snímku (taktu herní smyčky):
      1. Seberou se senzory všech jednotek (blízkost nepřítele, zdraví, náboje, lékárnička).
      2. Senzory se pošlou jako matice do tvé RTX 5070 Ti.
      3. Tensor Blob (268 bajtů) na GPU bleskově rozhodne o akcích všech jednotek naráz:
           ⚔️ ÚTOK  (červená)
           🏃 ÚTĚK  (žlutá)
           💊 LÉČIT (zelená)
      4. Aréna se překreslí přímo v terminálu.

Spuštění:
  python arena_sim.py
"""

import os
import sys
import time
import random
import torch
import numpy as np
from brick import Brick

# Herní pravidla logiky (pravdivostní tabulka pro boty)
def game_ai_rules(s0_enemy, s1_low_hp, s2_has_medkit, s3_has_ammo):
    attack = int(s0_enemy and s3_has_ammo and not s1_low_hp)
    flee = int(s0_enemy and (not s3_has_ammo or s1_low_hp))
    heal = int(s1_low_hp and s2_has_medkit)
    return (attack, flee, heal)

def get_trained_weights():
    # Rychlý trénink malého blobu (11 neuronů, 67 parametrů)
    brick = Brick.learn("ARENA_BLOB", 4, 3, game_ai_rules, hidden=[8], max_epochs=4000, lr=0.08)
    return brick.params

class Unit:
    def __init__(self, uid, x, y, team):
        self.uid = uid
        self.x = x
        self.y = y
        self.team = team # 0 = Modří (A), 1 = Červení (B)
        self.hp = 100
        self.ammo = random.randint(2, 6)
        self.medkits = random.choice([0, 1])
        self.last_action = "STÁT"

def run_game_loop():
    WIDTH, HEIGHT = 40, 16
    NUM_UNITS = 30

    print("Inicializuji Tensor Blob na RTX 5070 Ti...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = get_trained_weights()

    W1 = torch.tensor(params[0][0], dtype=torch.float32, device=device)
    b1 = torch.tensor(params[0][1], dtype=torch.float32, device=device)
    W2 = torch.tensor(params[1][0], dtype=torch.float32, device=device)
    b2 = torch.tensor(params[1][1], dtype=torch.float32, device=device)

    # Vytvoření jednotek
    units = []
    for i in range(NUM_UNITS):
        team = i % 2
        ux = random.randint(1, WIDTH - 2)
        uy = random.randint(1, HEIGHT - 2)
        units.append(Unit(i, ux, uy, team))

    print(f"Jednotky vytvořeny. Spouštím herní smyčku s RTX mozkem... (Ukončení: Ctrl+C)")
    time.sleep(1.5)

    frame = 0
    try:
        while True:
            frame += 1
            t_frame_start = time.perf_counter()

            # -------------------------------------------------------------
            # KROK 1: Vnímání světa (Sběr senzorů pro všechny jednotky)
            # -------------------------------------------------------------
            sensor_matrix = []
            for u in units:
                if u.hp <= 0:
                    sensor_matrix.append([0, 0, 0, 0])
                    continue
                # Hledáme nejbližšího živého nepřítele
                closest_dist = 999
                for other in units:
                    if other.team != u.team and other.hp > 0:
                        dist = abs(u.x - other.x) + abs(u.y - other.y)
                        if dist < closest_dist:
                            closest_dist = dist
                
                s0_enemy_near = 1 if closest_dist <= 3 else 0
                s1_low_hp = 1 if u.hp <= 35 else 0
                s2_has_medkit = 1 if u.medkits > 0 else 0
                s3_has_ammo = 1 if u.ammo > 0 else 0

                sensor_matrix.append([s0_enemy_near, s1_low_hp, s2_has_medkit, s3_has_ammo])

            # -------------------------------------------------------------
            # KROK 2: Bleskový výpočet rozhodnutí na RTX 5070 Ti (CUDA)
            # -------------------------------------------------------------
            t_gpu_start = time.perf_counter()
            sensors_gpu = torch.tensor(sensor_matrix, dtype=torch.float32, device=device)
            
            # Průlet signálu přes tenzorový cucek vah
            h = torch.tanh(sensors_gpu @ W1 + b1)
            actions_gpu = (torch.sigmoid(h @ W2 + b2) >= 0.5).to(torch.int8)
            torch.cuda.synchronize()
            gpu_time_us = (time.perf_counter() - t_gpu_start) * 1_000_000 # v mikrosekundách!
            
            decisions = actions_gpu.cpu().numpy()

            # -------------------------------------------------------------
            # KROK 3: Provedení herních akcí ve světě
            # -------------------------------------------------------------
            for i, u in enumerate(units):
                if u.hp <= 0:
                    continue
                atk, flee, heal = decisions[i]
                
                if heal:
                    u.hp = min(100, u.hp + 40)
                    u.medkits -= 1
                    u.last_action = "LÉČIT (+HP)"
                elif atk:
                    u.ammo -= 1
                    u.last_action = "ÚTOK! (Palba)"
                    # Zranit náhodného nepřítele v dosahu
                    for other in units:
                        if other.team != u.team and other.hp > 0 and (abs(u.x - other.x) + abs(u.y - other.y) <= 3):
                            other.hp -= 25
                            break
                elif flee:
                    # Krok směrem od středu
                    u.x = max(1, min(WIDTH - 2, u.x + random.choice([-1, 1])))
                    u.y = max(1, min(HEIGHT - 2, u.y + random.choice([-1, 1])))
                    u.last_action = "ÚTĚK!"
                else:
                    # Klidné bloumání po aréně
                    u.x = max(1, min(WIDTH - 2, u.x + random.choice([-1, 0, 1])))
                    u.y = max(1, min(HEIGHT - 2, u.y + random.choice([-1, 0, 1])))
                    u.last_action = "HLÍDKA"

            # -------------------------------------------------------------
            # KROK 4: Vykreslení arény do terminálu (Frame buffer)
            # -------------------------------------------------------------
            grid = [[" " for _ in range(WIDTH)] for _ in range(HEIGHT)]
            # Ohraničení arény
            for x in range(WIDTH):
                grid[0][x] = "─"
                grid[HEIGHT - 1][x] = "─"
            for y in range(HEIGHT):
                grid[y][0] = "│"
                grid[y][WIDTH - 1] = "│"

            team_a_alive = sum(1 for u in units if u.team == 0 and u.hp > 0)
            team_b_alive = sum(1 for u in units if u.team == 1 and u.hp > 0)

            for u in units:
                if u.hp > 0:
                    icon = "A" if u.team == 0 else "B"
                    grid[u.y][u.x] = icon
                else:
                    grid[u.y][u.x] = "x" # Padlý bot

            # Sestavení obrazovky
            lines = []
            lines.append(f"┌{'─' * (WIDTH - 2)}┐")
            for y in range(1, HEIGHT - 1):
                lines.append("│" + "".join(grid[y][1:WIDTH - 1]) + "│")
            lines.append(f"└{'─' * (WIDTH - 2)}┘")

            screen = "\n".join(lines)
            
            # Vyčištění a tisk
            os.system("cls" if os.name == "nt" else "clear")
            print("=" * 60)
            print(" 🎮 TENSOR BLOB V HERNÍ SMYČCE (RTX 5070 Ti)")
            print("=" * 60)
            print(screen)
            print(f" Snímek #{frame:04d} | Živí: Tým A={team_a_alive} (A), Tým B={team_b_alive} (B)")
            print(f" ⏱️ Čas Tensor Blobu na RTX: {gpu_time_us:.1f} µs (mikrosekund!)")
            print("------------------------------------------------------------")
            print(" Stav prvních 3 botů v reálném čase:")
            for u in units[:3]:
                stav = f"HP={u.hp:3d} | Ammo={u.ammo} | Med={u.medkits} -> {u.last_action}" if u.hp > 0 else "MRTEV"
                team_tag = "Tým A" if u.team == 0 else "Tým B"
                print(f"  Bot #{u.uid:02d} ({team_tag}): {stav}")
            print("=" * 60)
            print(" Pro ukončení stiskni Ctrl+C")

            # Drobná pauza, aby lidské oko hru stíhalo sledovat (cca 5 FPS pro zrak)
            time.sleep(0.2)

            if team_a_alive == 0 or team_b_alive == 0:
                winner = "TÝM A (Modří)" if team_a_alive > 0 else "TÝM B (Červení)"
                print(f"\n🏆 KONEC BITVY! Zvítězil {winner} po {frame} snímcích!")
                break

    except KeyboardInterrupt:
        print("\nHerní smyčka zastavena hráčem.")

if __name__ == "__main__":
    run_game_loop()
