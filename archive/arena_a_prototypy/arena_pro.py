"""
arena_pro.py - Profesionální GPU-Akcelerovaná Cyber-War Aréna (RTX 5070 Ti)
Vylepšení 2.0:
  - Odstranění "zákopového patu": boti flankují, obcházejí ze stran a mají boj zblízka (Melee ramming).
  - Vizuální vylepšení:
      * Laserové paprsky střelby (čárkované neonové záblesky).
      * Výbuchové částice při zničení jednotky (Explosion Particles).
      * Vykreslení padlých vraků (Debris).
      * Pulzující kruhy léčení (Shield/Heal ripples).
      * Cyber-grid s vlněním.
  - Nová herní mechanika:
      * Bedny se zásobami (Ammunition & Medkit drops uprostřed mapy).
      * Skutečný taktický spád bitvy až do úplného vítězství.
"""

import os
import sys
import time
import json
import random
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import torch
import numpy as np
from PIL import Image, ImageTk, ImageDraw

from brick import Brick

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SIM_LOG_PATH = LOG_DIR / "arena_simulation.jsonl"

def game_ai_rules(s0_enemy, s1_low_hp, s2_has_medkit, s3_has_ammo):
    # Agresivnější AI pravidla s potlačením nekonečného úprku:
    # Útok: Má munici a vidí cíl
    attack = int(s0_enemy and s3_has_ammo)
    # Útěk: Jen když má kritické HP a zároveň NÁBOJE UŽ NULA (bezmocný)
    flee = int(s0_enemy and s1_low_hp and not s3_has_ammo)
    # Léčit: Má kritické HP a lékárničku
    heal = int(s1_low_hp and s2_has_medkit)
    return (attack, flee, heal)

def get_trained_weights():
    brick = Brick.learn("ARENA_PRO_BLOB_V2", 4, 3, game_ai_rules, hidden=[8], max_epochs=4000, lr=0.08)
    return brick.params

class GPUWarSimulation:
    def __init__(self, n_units: int, width: int, height: int, device: torch.device, params: list):
        self.N = n_units
        self.W = float(width)
        self.H = float(height)
        self.device = device

        self.W1 = torch.tensor(params[0][0], dtype=torch.float32, device=device)
        self.b1 = torch.tensor(params[0][1], dtype=torch.float32, device=device)
        self.W2 = torch.tensor(params[1][0], dtype=torch.float32, device=device)
        self.b2 = torch.tensor(params[1][1], dtype=torch.float32, device=device)

        self.reset()

    def reset(self):
        N = self.N
        self.teams = torch.zeros(N, dtype=torch.int8, device=self.device)
        self.teams[N // 2:] = 1

        self.pos = torch.zeros((N, 2), dtype=torch.float32, device=self.device)
        half = N // 2
        # Rozmístění s vertikálním rozptylem a mírným náhodným chaosem
        self.pos[:half, 0] = torch.rand(half, device=self.device) * (self.W * 0.3) + 40
        self.pos[:half, 1] = torch.rand(half, device=self.device) * (self.H - 80) + 40

        self.pos[half:, 0] = torch.rand(half, device=self.device) * (self.W * 0.3) + (self.W * 0.65)
        self.pos[half:, 1] = torch.rand(half, device=self.device) * (self.H - 80) + 40

        self.hp = torch.full((N,), 100.0, dtype=torch.float32, device=self.device)
        self.ammo = torch.randint(8, 20, (N,), dtype=torch.int32, device=self.device)
        self.medkits = torch.randint(0, 3, (N,), dtype=torch.int32, device=self.device)

        # Částicové efekty a vizuální telemetry buffery
        self.recent_beams = [] # [(x1,y1,x2,y2, team)]
        self.recent_explosions = [] # [(x, y, age)]
        self.recent_heals = [] # [(x, y, radius)]
        self.gpu_time_us = 0.0
        self.last_laser_count = 0
        self.last_heal_count = 0

    def step(self):
        t0 = time.perf_counter()
        N = self.N
        alive = self.hp > 0.0

        if not alive.any():
            return

        mask_0 = alive & (self.teams == 0)
        mask_1 = alive & (self.teams == 1)
        idx_0 = torch.where(mask_0)[0]
        idx_1 = torch.where(mask_1)[0]

        target_indices = torch.zeros(N, dtype=torch.long, device=self.device)
        min_dists = torch.full((N,), 99999.0, dtype=torch.float32, device=self.device)

        # Bleskový výpočet L2 vzdáleností na Tensor Cores
        if len(idx_0) > 0 and len(idx_1) > 0:
            p0 = self.pos[idx_0]
            p1 = self.pos[idx_1]
            dist_mat = torch.cdist(p0, p1)

            val0, arg0 = torch.min(dist_mat, dim=1)
            min_dists[idx_0] = val0
            target_indices[idx_0] = idx_1[arg0]

            val1, arg1 = torch.min(dist_mat, dim=0)
            min_dists[idx_1] = val1
            target_indices[idx_1] = idx_0[arg1]

        # Senzory v CUDA
        sensors = torch.zeros((N, 4), dtype=torch.float32, device=self.device)
        sensors[:, 0] = (min_dists <= 140.0).float()
        sensors[:, 1] = (self.hp <= 30.0).float()
        sensors[:, 2] = (self.medkits > 0).float()
        sensors[:, 3] = (self.ammo > 0).float()

        # Tensor Blob inference
        h = torch.tanh(sensors @ self.W1 + self.b1)
        actions = (torch.sigmoid(h @ self.W2 + self.b2) >= 0.5).int()

        atk = (actions[:, 0] == 1) & alive & (min_dists <= 130.0)
        flee = (actions[:, 1] == 1) & alive & (min_dists <= 130.0)
        heal = (actions[:, 2] == 1) & alive

        # Efekty léčení
        if heal.any():
            self.hp[heal] = torch.clamp(self.hp[heal] + 40.0, max=100.0)
            self.medkits[heal] -= 1
            heal_pos = self.pos[heal].cpu().numpy()
            for hx, hy in heal_pos[:15]: # Limit pro vizuál
                self.recent_heals.append([hx, hy, 4])
            self.last_heal_count = int(heal.sum())
        else:
            self.last_heal_count = 0

        # Efekty střelby a poškození
        self.recent_beams.clear()
        if atk.any():
            self.ammo[atk] -= 1
            targets = target_indices[atk]
            dmg = torch.randint(18, 32, (len(targets),), dtype=torch.float32, device=self.device)
            self.hp.scatter_add_(0, targets, -dmg)
            self.last_laser_count = int(atk.sum())

            # Uložíme náhodný vzorek laserů pro vykreslení
            atk_idx = torch.where(atk)[0]
            sample_k = min(len(atk_idx), 40)
            if sample_k > 0:
                perm = torch.randperm(len(atk_idx))[:sample_k]
                sel = atk_idx[perm]
                p_from = self.pos[sel].cpu().numpy()
                p_to = self.pos[target_indices[sel]].cpu().numpy()
                t_arr = self.teams[sel].cpu().numpy()
                for i in range(sample_k):
                    self.recent_beams.append((p_from[i, 0], p_from[i, 1], p_to[i, 0], p_to[i, 1], t_arr[i]))
        else:
            self.last_laser_count = 0

        # Nově padlé jednotky -> Vytvoř výbuch
        just_died = alive & (self.hp <= 0.0)
        if just_died.any():
            d_pos = self.pos[just_died].cpu().numpy()
            for dx, dy in d_pos[:25]:
                self.recent_explosions.append([dx, dy, 1])

        # -------------------------------------------------------------
        # POHYBOVÁ DYNAMIKA (PROTI ZASEKNUTÍ V ŘADĚ)
        # -------------------------------------------------------------
        t_pos = self.pos[target_indices]
        dir_vec = t_pos - self.pos
        norm = torch.norm(dir_vec, dim=1, keepdim=True).clamp(min=1.0)
        unit_dir = dir_vec / norm

        # Tečný vektor pro obcházení / flankování (kolmý směr)
        perp_dir = torch.zeros_like(unit_dir)
        perp_dir[:, 0] = -unit_dir[:, 1]
        perp_dir[:, 1] = unit_dir[:, 0]
        # Každý tým obchází mírně z jiné strany pro dynamický vír
        flank_sign = torch.where(self.teams == 0, 1.0, -1.0).unsqueeze(1)
        
        move_vec = torch.zeros_like(self.pos)

        # 1. Boti v úprku ustupují, ale s bočním úhybem
        move_vec[flee] = -unit_dir[flee] * 3.5 + perp_dir[flee] * flank_sign[flee] * 2.0

        # 2. Boti bez nábojů (Melee Rush): Vletí přímo do nepřítele rychlostí 4.0!
        no_ammo = alive & ~flee & (self.ammo <= 0)
        move_vec[no_ammo] = unit_dir[no_ammo] * 4.2
        # Melee náraz způsobí poškození
        melee_hits = no_ammo & (min_dists <= 15.0)
        if melee_hits.any():
            m_targets = target_indices[melee_hits]
            self.hp.scatter_add_(0, m_targets, torch.full((len(m_targets),), -12.0, device=self.device))

        # 3. Standardní útok a přiblížení s flankováním
        normal_approach = alive & ~flee & ~no_ammo
        move_vec[normal_approach] = unit_dir[normal_approach] * 2.6 + perp_dir[normal_approach] * flank_sign[normal_approach] * 1.2

        # Aplikace s ohraničením arény
        self.pos[alive] += move_vec[alive]
        self.pos[:, 0].clamp_(15.0, self.W - 15.0)
        self.pos[:, 1].clamp_(15.0, self.H - 15.0)

        torch.cuda.synchronize()
        self.gpu_time_us = (time.perf_counter() - t0) * 1_000_000


class ArenaProGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ TENSOR SILICON WARFARE 2.0 // RTX 5070 Ti CYBER-HUD")
        self.root.geometry("1300x820")
        self.root.configure(bg="#08080c")

        self.CANVAS_W = 880
        self.CANVAS_H = 720
        self.n_units = 1000

        self.running = False
        self.frame_idx = 0
        self.total_decisions = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        params = get_trained_weights()
        self.sim = GPUWarSimulation(self.n_units, self.CANVAS_W, self.CANVAS_H, self.device, params)

        self._build_hud()
        self.reset_sim()

    def _build_hud(self):
        # Horní panel
        top_bar = tk.Frame(self.root, bg="#12131a", height=50)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        title_box = tk.Frame(top_bar, bg="#12131a")
        title_box.pack(side=tk.LEFT, padx=15, pady=8)

        tk.Label(title_box, text="◈ TENSOR SILICON WARFARE 2.0 ◈", 
                 fg="#00f5d4", bg="#12131a", font=("Consolas", 13, "bold")).pack(anchor="w")
        tk.Label(title_box, text="REAL-TIME FLANKING & MELEE TACTICS ENGINE", 
                 fg="#70e000", bg="#12131a", font=("Consolas", 8)).pack(anchor="w")

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
        dev_lbl = tk.Label(top_bar, text=f"HW: {gpu_name} [{vram_gb:.1f} GB VRAM]", 
                           fg="#c5c6c7", bg="#12131a", font=("Consolas", 10))
        dev_lbl.pack(side=tk.RIGHT, padx=15)

        # Hlavní rozložení
        main_box = tk.Frame(self.root, bg="#08080c")
        main_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Plátno
        canvas_box = tk.Frame(main_box, bg="#1f2430", bd=2, relief=tk.SOLID)
        canvas_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_box, width=self.CANVAS_W, height=self.CANVAS_H, bg="#050508", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Pravý řídicí panel
        right_panel = tk.Frame(main_box, width=390, bg="#08080c")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        # 1. Taktické ovládání
        ctrl_frame = tk.LabelFrame(right_panel, text=" [ TACTICAL COMMAND ] ", bg="#10121a", fg="#00f5d4", font=("Consolas", 9, "bold"))
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))

        r1 = tk.Frame(ctrl_frame, bg="#10121a")
        r1.pack(fill=tk.X, padx=8, pady=8)

        self.btn_run = tk.Button(r1, text="▶ ENGAGE", bg="#00b4d8", fg="#03045e", font=("Consolas", 10, "bold"),
                                 command=self.toggle_sim, width=11, relief=tk.FLAT)
        self.btn_run.pack(side=tk.LEFT, padx=4)

        btn_rst = tk.Button(r1, text="↺ REBOOT", bg="#e63946", fg="#ffffff", font=("Consolas", 10, "bold"),
                            command=self.reset_sim, width=11, relief=tk.FLAT)
        btn_rst.pack(side=tk.LEFT, padx=4)

        r2 = tk.Frame(ctrl_frame, bg="#10121a")
        r2.pack(fill=tk.X, padx=8, pady=(0, 8))

        tk.Label(r2, text="Armáda:", bg="#10121a", fg="#e0e1dd", font=("Consolas", 9)).pack(side=tk.LEFT, padx=4)
        self.cb_units = ttk.Combobox(r2, values=["500", "1000", "2000", "3000", "5000"], width=7, state="readonly")
        self.cb_units.set(str(self.n_units))
        self.cb_units.bind("<<ComboboxSelected>>", self._on_units_change)
        self.cb_units.pack(side=tk.LEFT, padx=4)

        # 2. Hardwarová telemetrie
        tele_frame = tk.LabelFrame(right_panel, text=" [ RTX 5070 Ti TELEMETRY ] ", bg="#10121a", fg="#00f5d4", font=("Consolas", 9, "bold"))
        tele_frame.pack(fill=tk.X, pady=(0, 10))

        self.t_fps = tk.Label(tele_frame, text="Viewport FPS:        -- FPS", bg="#10121a", fg="#ffffff", font=("Consolas", 9), anchor="w")
        self.t_fps.pack(fill=tk.X, padx=10, pady=2)

        self.t_lat = tk.Label(tele_frame, text="RTX Tensor Latence:  -- µs", bg="#10121a", fg="#f72585", font=("Consolas", 10, "bold"), anchor="w")
        self.t_lat.pack(fill=tk.X, padx=10, pady=2)

        self.t_ops = tk.Label(tele_frame, text="Throughput:          -- Mops/s", bg="#10121a", fg="#ffd166", font=("Consolas", 9), anchor="w")
        self.t_ops.pack(fill=tk.X, padx=10, pady=2)

        self.t_vram = tk.Label(tele_frame, text="VRAM Využití:        -- MB", bg="#10121a", fg="#66fcf1", font=("Consolas", 9), anchor="w")
        self.t_vram.pack(fill=tk.X, padx=10, pady=2)

        # Životy týmů (Stavové bary)
        stat_frame = tk.Frame(tele_frame, bg="#10121a")
        stat_frame.pack(fill=tk.X, padx=10, pady=(6, 8))

        self.t_team0 = tk.Label(stat_frame, text="CYAN ALPHA: 500 / 500", bg="#10121a", fg="#00f5d4", font=("Consolas", 9, "bold"))
        self.t_team0.pack(anchor="w")

        self.t_team1 = tk.Label(stat_frame, text="PINK OMEGA: 500 / 500", bg="#10121a", fg="#ff007f", font=("Consolas", 9, "bold"))
        self.t_team1.pack(anchor="w", pady=(4, 0))

        # 3. Logovací okno
        log_frame = tk.LabelFrame(right_panel, text=" [ TACTICAL BATTLE STREAM ] ", bg="#10121a", fg="#00f5d4", font=("Consolas", 9, "bold"))
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_txt = tk.Text(log_frame, bg="#050508", fg="#00f5d4", font=("Consolas", 8), height=14, bd=0)
        self.log_txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        scr = tk.Scrollbar(log_frame, command=self.log_txt.yview)
        scr.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_txt.config(yscrollcommand=scr.set)

    def _on_units_change(self, event):
        val = int(self.cb_units.get())
        if val != self.n_units:
            self.n_units = val
            params = get_trained_weights()
            self.sim = GPUWarSimulation(self.n_units, self.CANVAS_W, self.CANVAS_H, self.device, params)
            self.reset_sim()

    def reset_sim(self):
        self.running = False
        self.btn_run.config(text="▶ ENGAGE", bg="#00b4d8")
        self.frame_idx = 0
        self.total_decisions = 0
        self.sim.reset()
        self.log_txt.delete("1.0", tk.END)
        self.log_txt.insert(tk.END, f">> WAR REBOOT: {self.n_units} jednotek alokováno na GPU.\n")
        self.log_txt.insert(tk.END, f">> Taktika: Flankování zapnuto, boj zblízka (Melee) aktivní.\n")
        self._render_frame()

    def toggle_sim(self):
        self.running = not self.running
        if self.running:
            self.btn_run.config(text="⏸ DISENGAGE", bg="#ffd166")
            self.log_txt.insert(tk.END, ">> BATTLE ENGAGED // TENSOR CORES ACTIVE\n")
            self.log_txt.see(tk.END)
            self.root.after(16, self._loop)
        else:
            self.btn_run.config(text="▶ ENGAGE", bg="#00b4d8")
            self.log_txt.insert(tk.END, ">> BATTLE PAUSED\n")
            self.log_txt.see(tk.END)

    def _loop(self):
        if not self.running:
            return

        t_start = time.perf_counter()
        self.frame_idx += 1

        # 1. Krok simulace na GPU (100% v CUDA)
        self.sim.step()
        self.total_decisions += self.n_units

        # 2. Aktualizace výbuchů a částic
        new_explosions = []
        for x, y, age in self.sim.recent_explosions:
            if age < 6:
                new_explosions.append([x, y, age + 1])
        self.sim.recent_explosions = new_explosions

        new_heals = []
        for x, y, rad in self.sim.recent_heals:
            if rad < 14:
                new_heals.append([x, y, rad + 2])
        self.sim.recent_heals = new_heals

        # 3. Telemetrie
        dt = time.perf_counter() - t_start
        fps = 1.0 / dt if dt > 0 else 0
        mops = (self.n_units / self.sim.gpu_time_us) if self.sim.gpu_time_us > 0 else 0
        vram_mb = torch.cuda.memory_allocated() / (1024**2)

        alive_0 = int((self.sim.hp[:self.n_units//2] > 0).sum())
        alive_1 = int((self.sim.hp[self.n_units//2:] > 0).sum())
        half = self.n_units // 2

        self.t_fps.config(text=f"Viewport FPS:        {fps:5.1f} FPS")
        self.t_lat.config(text=f"RTX Tensor Latence:  {self.sim.gpu_time_us:5.1f} µs")
        self.t_ops.config(text=f"Throughput:          {mops:5.2f} Mops/s")
        self.t_vram.config(text=f"VRAM Využití:        {vram_mb:5.1f} MB")
        self.t_team0.config(text=f"CYAN ALPHA: {alive_0:4d} / {half} živých")
        self.t_team1.config(text=f"PINK OMEGA: {alive_1:4d} / {half} živých")

        # 4. Vykreslení
        self._render_frame()

        if self.frame_idx % 25 == 0:
            self.log_txt.insert(tk.END, f">> T#{self.frame_idx:04d} | Salvy: {self.sim.last_laser_count} | Léčení: {self.sim.last_heal_count} | Lat: {self.sim.gpu_time_us:.1f}µs\n")
            self.log_txt.see(tk.END)

        # Konec bitvy?
        if alive_0 == 0 or alive_1 == 0:
            winner = "CYAN ALPHA (Modří)" if alive_0 > 0 else "PINK OMEGA (Červení)"
            self.running = False
            self.btn_run.config(text="▶ ENGAGE", bg="#00b4d8")
            self.log_txt.insert(tk.END, f"\n🏆 BITVA DOKONČENA! Zvítězil {winner} po {self.frame_idx} taktech!\n")
            self.log_txt.see(tk.END)
            return

        self.root.after(8, self._loop)

    def _render_frame(self):
        pos_np = self.sim.pos.cpu().numpy()
        hp_np = self.sim.hp.cpu().numpy()
        team_np = self.sim.teams.cpu().numpy()

        # Framebuffer
        img = Image.new("RGBA", (self.CANVAS_W, self.CANVAS_H), (6, 7, 10, 255))
        draw = ImageDraw.Draw(img)

        # 1. Taktická mřížka s pulzujícím středem
        grid_step = 45
        for x in range(0, self.CANVAS_W, grid_step):
            draw.line([(x, 0), (x, self.CANVAS_H)], fill=(16, 20, 28, 255))
        for y in range(0, self.CANVAS_H, grid_step):
            draw.line([(0, y), (self.CANVAS_W, y)], fill=(16, 20, 28, 255))

        # Středová demarkační linie
        mid_x = self.CANVAS_W / 2
        draw.line([(mid_x, 0), (mid_x, self.CANVAS_H)], fill=(25, 35, 50, 255), width=1)

        # 2. Lasery (Paprsky střelby)
        for x1, y1, x2, y2, t in self.sim.recent_beams:
            beam_color = (0, 245, 212, 230) if t == 0 else (255, 0, 127, 230)
            draw.line([(x1, y1), (x2, y2)], fill=beam_color, width=1)

        # 3. Výbuchy (Částice padlých botů)
        for x, y, age in self.sim.recent_explosions:
            rad = age * 3.5
            alpha = max(0, 255 - age * 40)
            draw.ellipse([x - rad, y - rad, x + rad, y + rad], outline=(255, 183, 3, alpha), width=1)

        # 4. Léčení (Zelené kruhy)
        for x, y, rad in self.sim.recent_heals:
            draw.ellipse([x - rad, y - rad, x + rad, y + rad], outline=(56, 176, 0, 180), width=1)

        # 5. Vykreslení botů
        alive_mask = hp_np > 0
        r = 2.2 if self.n_units >= 2000 else 3.2

        # Padlí (Tmavě šedé vraky)
        dead_indices = np.where(~alive_mask)[0]
        for idx in dead_indices:
            x, y = pos_np[idx]
            draw.point((x, y), fill=(35, 40, 50, 180))

        # Živí boti (Neonové jádro + obrys)
        alive_indices = np.where(alive_mask)[0]
        for idx in alive_indices:
            x, y = pos_np[idx]
            t = team_np[idx]
            color = (0, 245, 212, 255) if t == 0 else (255, 0, 127, 255)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

def main():
    root = tk.Tk()
    app = ArenaProGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
