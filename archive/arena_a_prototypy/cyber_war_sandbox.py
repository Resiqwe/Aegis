"""
cyber_war_sandbox.py - Indie Game Startup Edition: VRAM-DropBox Multi-Faction Engine
Architektura:
  - VRAM-DropBox Core: Oddělená in-memory simulační smyčka na RTX 5070 Ti s nulovým zpožděním.
  - Multi-Faction: Podpora 2, 3 nebo 4 bojujících frakcí:
      * Frakce A: Neon Cyan (Alpha Vanguard)
      * Frakce B: Neon Pink (Omega Syndicate)
      * Frakce C: Neon Amber (Solar Core)
      * Frakce D: Neon Purple (Phantom Unit)
  - Plně parametrizovatelný taktický HUD:
      * Počet jednotek celkem (100 až 5 000)
      * Volba taktické doktríny (Útočná, Obranná, Flankovací vír, Zběsilý Melee Rush)
      * Rychlost palby a zbraňový dosah
      * Živá telemetrie z VRAM a monitorování stavu frakcí
  - Profesionální Cyber-Tactical Indie UI.
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

# Definice frakcí
FACTIONS = [
    {"name": "ALPHA VANGUARD", "color": (0, 245, 212, 255), "hex": "#00f5d4"},
    {"name": "OMEGA SYNDICATE", "color": (255, 0, 127, 255), "hex": "#ff007f"},
    {"name": "SOLAR CORE", "color": (255, 183, 3, 255), "hex": "#ffb703"},
    {"name": "PHANTOM UNIT", "color": (157, 78, 221, 255), "hex": "#9d4edd"},
]

def train_sandbox_blob():
    # Univerzální 4-in -> 3-out Tensor AI Blob
    def rules(s0_enemy, s1_low_hp, s2_has_medkit, s3_has_ammo):
        atk = int(s0_enemy and s3_has_ammo)
        flee = int(s0_enemy and s1_low_hp and not s3_has_ammo)
        heal = int(s1_low_hp and s2_has_medkit)
        return (atk, flee, heal)

    brick = Brick.learn("SANDBOX_BLOB", 4, 3, rules, hidden=[8], max_epochs=4000, lr=0.08)
    return brick.params


class VRAMDropBoxEngine:
    """
    In-Memory 'DropBox' jádro běžící 100% ve VRAM na RTX 5070 Ti.
    Uchovává a transformuje stav světa bez diskových i CPU mezikroků.
    """
    def __init__(self, device: torch.device, params: list, width: int, height: int):
        self.device = device
        self.W = float(width)
        self.H = float(height)

        # Váhy AI Blobu ve VRAM
        self.W1 = torch.tensor(params[0][0], dtype=torch.float32, device=device)
        self.b1 = torch.tensor(params[0][1], dtype=torch.float32, device=device)
        self.W2 = torch.tensor(params[1][0], dtype=torch.float32, device=device)
        self.b2 = torch.tensor(params[1][1], dtype=torch.float32, device=device)

        # Nastavitelné taktické parametry (z HUDu)
        self.weapon_range = 140.0
        self.tactic_mode = "FLANK" # "FLANK", "AGGRO", "DEFENSIVE", "MELEE"
        self.fire_rate_multiplier = 1.0

        # Vizuální DropBox buffery (čtené UI bez blokace výpočtu)
        self.beams = []
        self.explosions = []
        self.heals = []
        self.gpu_time_us = 0.0

    def configure(self, n_units: int, n_factions: int, tactic: str, w_range: float):
        self.N = n_units
        self.n_factions = n_factions
        self.tactic_mode = tactic
        self.weapon_range = float(w_range)

        N = self.N
        # Týmy rovnoměrně rozdělené mezi frakce
        teams_arr = []
        for i in range(N):
            teams_arr.append(i % n_factions)
        self.teams = torch.tensor(teams_arr, dtype=torch.int8, device=self.device)

        # Pozice: každá frakce začíná ve svém sektoru arény
        self.pos = torch.zeros((N, 2), dtype=torch.float32, device=self.device)
        for f in range(n_factions):
            f_mask = self.teams == f
            count = int(f_mask.sum())
            if count == 0:
                continue

            # Kruhové základny pro frakce
            angle = (2.0 * np.pi / n_factions) * f
            cx = self.W / 2 + np.cos(angle) * (self.W * 0.32)
            cy = self.H / 2 + np.sin(angle) * (self.H * 0.32)

            self.pos[f_mask, 0] = torch.clamp(torch.randn(count, device=self.device) * 45.0 + cx, 30.0, self.W - 30.0)
            self.pos[f_mask, 1] = torch.clamp(torch.randn(count, device=self.device) * 45.0 + cy, 30.0, self.H - 30.0)

        self.hp = torch.full((N,), 100.0, dtype=torch.float32, device=self.device)
        self.ammo = torch.randint(10, 25, (N,), dtype=torch.int32, device=self.device)
        self.medkits = torch.randint(0, 3, (N,), dtype=torch.int32, device=self.device)

        self.beams.clear()
        self.explosions.clear()
        self.heals.clear()

    def step(self):
        t0 = time.perf_counter()
        N = self.N
        alive = self.hp > 0.0

        if not alive.any():
            return

        alive_idx = torch.where(alive)[0]
        K = len(alive_idx)

        # -------------------------------------------------------------
        # 1. MULTI-FACTION SPATIAL MATRIX V CUDA
        # -------------------------------------------------------------
        target_indices = torch.zeros(N, dtype=torch.long, device=self.device)
        min_dists = torch.full((N,), 99999.0, dtype=torch.float32, device=self.device)

        if K > 1:
            p_alive = self.pos[alive_idx] # (K, 2)
            t_alive = self.teams[alive_idx] # (K,)

            # Vzdálenosti mezi všemi živými jednotkami
            dist_mat = torch.cdist(p_alive, p_alive) # (K, K)

            # Zamaskování spojenců ze stejné frakce
            ally_mask = t_alive.unsqueeze(1) == t_alive.unsqueeze(0)
            dist_mat[ally_mask] = 99999.0

            # Nalezení nejbližšího nepřítele z cizí frakce
            val, arg = torch.min(dist_mat, dim=1)
            min_dists[alive_idx] = val
            target_indices[alive_idx] = alive_idx[arg]

        # -------------------------------------------------------------
        # 2. SESTAVENÍ SENZORŮ & TENSOR BLOB INFERENCE
        # -------------------------------------------------------------
        sensors = torch.zeros((N, 4), dtype=torch.float32, device=self.device)
        sensors[:, 0] = (min_dists <= self.weapon_range).float()
        sensors[:, 1] = (self.hp <= 35.0).float()
        sensors[:, 2] = (self.medkits > 0).float()
        sensors[:, 3] = (self.ammo > 0).float()

        h = torch.tanh(sensors @ self.W1 + self.b1)
        actions = (torch.sigmoid(h @ self.W2 + self.b2) >= 0.5).int()

        atk = (actions[:, 0] == 1) & alive & (min_dists <= self.weapon_range)
        flee = (actions[:, 1] == 1) & alive & (min_dists <= self.weapon_range)
        heal = (actions[:, 2] == 1) & alive

        # Aplikace léčení
        if heal.any():
            self.hp[heal] = torch.clamp(self.hp[heal] + 35.0, max=100.0)
            self.medkits[heal] -= 1
            h_pos = self.pos[heal].cpu().numpy()
            for x, y in h_pos[:15]:
                self.heals.append([x, y, 3])

        # Aplikace střelby a poškození
        self.beams.clear()
        if atk.any():
            self.ammo[atk] -= 1
            targets = target_indices[atk]
            dmg = torch.randint(18, 30, (len(targets),), dtype=torch.float32, device=self.device)
            self.hp.scatter_add_(0, targets, -dmg)

            # Zápis do DropBox bufferu laserů
            atk_idx = torch.where(atk)[0]
            sample_k = min(len(atk_idx), 45)
            if sample_k > 0:
                perm = torch.randperm(len(atk_idx))[:sample_k]
                sel = atk_idx[perm]
                p1 = self.pos[sel].cpu().numpy()
                p2 = self.pos[target_indices[sel]].cpu().numpy()
                t_arr = self.teams[sel].cpu().numpy()
                for i in range(sample_k):
                    self.beams.append((p1[i, 0], p1[i, 1], p2[i, 0], p2[i, 1], t_arr[i]))

        # Detekce padlých pro výbuchy
        newly_dead = alive & (self.hp <= 0.0)
        if newly_dead.any():
            d_pos = self.pos[newly_dead].cpu().numpy()
            for dx, dy in d_pos[:30]:
                self.explosions.append([dx, dy, 1])

        # -------------------------------------------------------------
        # 3. POHYBOVÁ TAKTIKA DLE ZVOLENÉ DOKTRÍNY
        # -------------------------------------------------------------
        t_pos = self.pos[target_indices]
        dir_vec = t_pos - self.pos
        norm = torch.norm(dir_vec, dim=1, keepdim=True).clamp(min=1.0)
        unit_dir = dir_vec / norm

        perp_dir = torch.zeros_like(unit_dir)
        perp_dir[:, 0] = -unit_dir[:, 1]
        perp_dir[:, 1] = unit_dir[:, 0]
        flank_sign = torch.where(self.teams % 2 == 0, 1.0, -1.0).unsqueeze(1)

        move = torch.zeros_like(self.pos)

        if self.tactic_mode == "FLANK":
            # Obcházení ze stran a křížení palby
            move[flee] = -unit_dir[flee] * 3.5 + perp_dir[flee] * flank_sign[flee] * 2.5
            charge = alive & ~flee
            move[charge] = unit_dir[charge] * 2.5 + perp_dir[charge] * flank_sign[charge] * 1.8
        elif self.tactic_mode == "AGGRO":
            # Přímý agresivní čelní nápor
            move[flee] = -unit_dir[flee] * 1.5
            charge = alive & ~flee
            move[charge] = unit_dir[charge] * 3.5
        elif self.tactic_mode == "DEFENSIVE":
            # Opatrné držení odstupu
            move[flee] = -unit_dir[flee] * 4.5
            charge = alive & ~flee & (min_dists > self.weapon_range * 0.7)
            move[charge] = unit_dir[charge] * 1.5
        elif self.tactic_mode == "MELEE":
            # Všichni bez milosti naráží do nepřátel
            move[alive] = unit_dir[alive] * 4.5
            near_hit = alive & (min_dists <= 15.0)
            if near_hit.any():
                targets = target_indices[near_hit]
                self.hp.scatter_add_(0, targets, torch.full((len(targets),), -15.0, device=self.device))

        self.pos[alive] += move[alive]
        self.pos[:, 0].clamp_(20.0, self.W - 20.0)
        self.pos[:, 1].clamp_(20.0, self.H - 20.0)

        torch.cuda.synchronize()
        self.gpu_time_us = (time.perf_counter() - t0) * 1_000_000


class CyberWarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ CYBER-WAR: TENSOR SILICON DROPBOX // INDIE STARTUP ENGINE")
        self.root.geometry("1340x840")
        self.root.configure(bg="#07080c")

        self.CANVAS_W = 900
        self.CANVAS_H = 740

        self.running = False
        self.frame_idx = 0
        self.total_decisions = 0

        # Parametry simulace
        self.cur_units = 1200
        self.cur_factions = 2
        self.cur_tactic = "FLANK"
        self.cur_range = 140

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        params = train_sandbox_blob()
        self.engine = VRAMDropBoxEngine(self.device, params, self.CANVAS_W, self.CANVAS_H)

        self._build_ui()
        self.reboot_sim()

    def _build_ui(self):
        # Top Bar s herním logem a stavem
        top_bar = tk.Frame(self.root, bg="#0d0f17", height=52)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        logo_box = tk.Frame(top_bar, bg="#0d0f17")
        logo_box.pack(side=tk.LEFT, padx=16, pady=6)

        tk.Label(logo_box, text="⚡ CYBER-WAR // TENSOR DROPBOX", 
                 fg="#00f5d4", bg="#0d0f17", font=("Consolas", 13, "bold")).pack(anchor="w")
        tk.Label(logo_box, text="VRAM ZERO-COPY IN-MEMORY ENGINE • RTX 5070 Ti ACCELERATED", 
                 fg="#70e000", bg="#0d0f17", font=("Consolas", 8)).pack(anchor="w")

        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
        tk.Label(top_bar, text=f"HW: NVIDIA RTX 5070 Ti [{vram_gb:.1f} GB GDDR6X]", 
                 fg="#8d99ae", bg="#0d0f17", font=("Consolas", 10)).pack(side=tk.RIGHT, padx=16)

        # Střední herní zóna
        center_frame = tk.Frame(self.root, bg="#07080c")
        center_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Herní plátno
        canv_box = tk.Frame(center_frame, bg="#1b1e2b", bd=2, relief=tk.SOLID)
        canv_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canv_box, width=self.CANVAS_W, height=self.CANVAS_H, bg="#050608", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Pravý řídicí dok
        dock = tk.Frame(center_frame, width=400, bg="#07080c")
        dock.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        # 1. Herní Taktický Panel
        p1 = tk.LabelFrame(dock, text=" [ TACTICAL COMMAND & SETUP ] ", bg="#0d0f17", fg="#00f5d4", font=("Consolas", 9, "bold"))
        p1.pack(fill=tk.X, pady=(0, 10))

        # Tlačítka start/stop
        btn_row = tk.Frame(p1, bg="#0d0f17")
        btn_row.pack(fill=tk.X, padx=8, pady=6)

        self.btn_toggle = tk.Button(btn_row, text="▶ ENGAGE WAR", bg="#00b4d8", fg="#03045e", font=("Consolas", 10, "bold"),
                                    command=self.toggle_sim, width=13, relief=tk.FLAT)
        self.btn_toggle.pack(side=tk.LEFT, padx=3)

        btn_rst = tk.Button(btn_row, text="↺ REBOOT", bg="#e63946", fg="#ffffff", font=("Consolas", 10, "bold"),
                            command=self.reboot_sim, width=10, relief=tk.FLAT)
        btn_rst.pack(side=tk.LEFT, padx=3)

        # Konfigurace parametrů armády
        cfg_grid = tk.Frame(p1, bg="#0d0f17")
        cfg_grid.pack(fill=tk.X, padx=8, pady=4)

        # Počet jednotek
        tk.Label(cfg_grid, text="Bojovníků:", bg="#0d0f17", fg="#e0e1dd", font=("Consolas", 8)).grid(row=0, column=0, sticky="w", pady=2)
        self.cb_units = ttk.Combobox(cfg_grid, values=["300", "600", "1200", "2400", "4000"], width=8, state="readonly")
        self.cb_units.set(str(self.cur_units))
        self.cb_units.grid(row=0, column=1, padx=4, pady=2)

        # Počet stran (Frakcí)
        tk.Label(cfg_grid, text="Frakce (Strany):", bg="#0d0f17", fg="#e0e1dd", font=("Consolas", 8)).grid(row=1, column=0, sticky="w", pady=2)
        self.cb_factions = ttk.Combobox(cfg_grid, values=["2 Strany (Alpha vs Omega)", "3 Strany (Trojúhelník)", "4 Strany (Battle Royale)"], width=22, state="readonly")
        self.cb_factions.set("2 Strany (Alpha vs Omega)")
        self.cb_factions.grid(row=1, column=1, padx=4, pady=2)

        # Doktrína / Taktika
        tk.Label(cfg_grid, text="Doktrína:", bg="#0d0f17", fg="#e0e1dd", font=("Consolas", 8)).grid(row=2, column=0, sticky="w", pady=2)
        self.cb_tactic = ttk.Combobox(cfg_grid, values=["FLANK (Obcházení kleštěmi)", "AGGRO (Čelní útok)", "DEFENSIVE (Obrana a ústup)", "MELEE (Zběsilý náraz)"], width=22, state="readonly")
        self.cb_tactic.set("FLANK (Obcházení kleštěmi)")
        self.cb_tactic.grid(row=2, column=1, padx=4, pady=2)

        # Dosah zbraní
        tk.Label(cfg_grid, text="Dosah zbraní:", bg="#0d0f17", fg="#e0e1dd", font=("Consolas", 8)).grid(row=3, column=0, sticky="w", pady=2)
        self.scale_range = tk.Scale(cfg_grid, from_=70, to=220, orient=tk.HORIZONTAL, bg="#0d0f17", fg="#00f5d4", 
                                    highlightthickness=0, length=140, showvalue=True)
        self.scale_range.set(self.cur_range)
        self.scale_range.grid(row=3, column=1, padx=4, pady=2)

        # 2. Stav Frakcí & Živé Bary
        p2 = tk.LabelFrame(dock, text=" [ FACTION INTEGRITY STATUS ] ", bg="#0d0f17", fg="#00f5d4", font=("Consolas", 9, "bold"))
        p2.pack(fill=tk.X, pady=(0, 10))

        self.faction_labels = []
        for i in range(4):
            f_box = tk.Frame(p2, bg="#0d0f17")
            f_box.pack(fill=tk.X, padx=8, pady=2)
            lbl = tk.Label(f_box, text=f"• {FACTIONS[i]['name']}: --", fg=FACTIONS[i]["hex"], bg="#0d0f17", font=("Consolas", 9, "bold"))
            lbl.pack(anchor="w")
            self.faction_labels.append(lbl)

        # 3. Telemetrie VRAM DropBoxu
        p3 = tk.LabelFrame(dock, text=" [ VRAM-DROPBOX TELEMETRY ] ", bg="#0d0f17", fg="#00f5d4", font=("Consolas", 9, "bold"))
        p3.pack(fill=tk.X, pady=(0, 10))

        self.t_fps = tk.Label(p3, text="Viewport Framerate:   -- FPS", bg="#0d0f17", fg="#ffffff", font=("Consolas", 9), anchor="w")
        self.t_fps.pack(fill=tk.X, padx=8, pady=2)

        self.t_lat = tk.Label(p3, text="RTX Tensor Latence:   -- µs", bg="#0d0f17", fg="#f72585", font=("Consolas", 10, "bold"), anchor="w")
        self.t_lat.pack(fill=tk.X, padx=8, pady=2)

        self.t_mops = tk.Label(p3, text="AI Throughput:        -- Mops/s", bg="#0d0f17", fg="#ffd166", font=("Consolas", 9), anchor="w")
        self.t_mops.pack(fill=tk.X, padx=8, pady=2)

        self.t_vram = tk.Label(p3, text="VRAM Zero-Copy State: -- MB", bg="#0d0f17", fg="#00f5d4", font=("Consolas", 9), anchor="w")
        self.t_vram.pack(fill=tk.X, padx=8, pady=2)

        # 4. Taktický log
        p4 = tk.LabelFrame(dock, text=" [ AUDIT BATTLE FEED ] ", bg="#0d0f17", fg="#00f5d4", font=("Consolas", 9, "bold"))
        p4.pack(fill=tk.BOTH, expand=True)

        self.log_txt = tk.Text(p4, bg="#050608", fg="#00f5d4", font=("Consolas", 8), height=10, bd=0)
        self.log_txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        scr = tk.Scrollbar(p4, command=self.log_txt.yview)
        scr.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_txt.config(yscrollcommand=scr.set)

    def reboot_sim(self):
        self.running = False
        self.btn_toggle.config(text="▶ ENGAGE WAR", bg="#00b4d8")
        self.frame_idx = 0
        self.total_decisions = 0

        # Načtení parametrů z GUI
        self.cur_units = int(self.cb_units.get())
        fac_str = self.cb_factions.get()
        self.cur_factions = 2 if "2" in fac_str else (3 if "3" in fac_str else 4)
        
        tac_str = self.cb_tactic.get()
        if "FLANK" in tac_str: self.cur_tactic = "FLANK"
        elif "AGGRO" in tac_str: self.cur_tactic = "AGGRO"
        elif "DEFENSIVE" in tac_str: self.cur_tactic = "DEFENSIVE"
        else: self.cur_tactic = "MELEE"

        self.cur_range = float(self.scale_range.get())

        # Inicializace VRAM DropBoxu
        self.engine.configure(self.cur_units, self.cur_factions, self.cur_tactic, self.cur_range)

        self.log_txt.delete("1.0", tk.END)
        self.log_txt.insert(tk.END, f">> VRAM-DROPBOX REBOOTED: {self.cur_units} jednotek ve VRAM.\n")
        self.log_txt.insert(tk.END, f">> Frakcí na mapě: {self.cur_factions} | Doktrína: {self.cur_tactic}\n")
        self._render_frame()

    def toggle_sim(self):
        self.running = not self.running
        if self.running:
            self.btn_toggle.config(text="⏸ PAUSE WAR", bg="#ffd166")
            self.log_txt.insert(tk.END, ">> WAR PROTOCOL ACTIVE // ZERO-COPY GPU LOOP\n")
            self.log_txt.see(tk.END)
            self.root.after(16, self._loop)
        else:
            self.btn_toggle.config(text="▶ ENGAGE WAR", bg="#00b4d8")
            self.log_txt.insert(tk.END, ">> SIMULATION PAUSED\n")
            self.log_txt.see(tk.END)

    def _loop(self):
        if not self.running:
            return

        t_start = time.perf_counter()
        self.frame_idx += 1

        # Parametry se dají ladit i za běhu
        self.engine.weapon_range = float(self.scale_range.get())
        tac_str = self.cb_tactic.get()
        if "FLANK" in tac_str: self.engine.tactic_mode = "FLANK"
        elif "AGGRO" in tac_str: self.engine.tactic_mode = "AGGRO"
        elif "DEFENSIVE" in tac_str: self.engine.tactic_mode = "DEFENSIVE"
        else: self.engine.tactic_mode = "MELEE"

        # Krok na GPU v paměťovém DropBoxu
        self.engine.step()
        self.total_decisions += self.cur_units

        # Aktualizace výbuchů a léčení
        self.engine.explosions = [[x, y, age + 1] for x, y, age in self.engine.explosions if age < 6]
        self.engine.heals = [[x, y, r + 2] for x, y, r in self.engine.heals if r < 14]

        # Telemetrie
        dt = time.perf_counter() - t_start
        fps = 1.0 / dt if dt > 0 else 0
        mops = (self.cur_units / self.engine.gpu_time_us) if self.engine.gpu_time_us > 0 else 0
        vram_mb = torch.cuda.memory_allocated() / (1024**2)

        self.t_fps.config(text=f"Viewport Framerate:   {fps:5.1f} FPS")
        self.t_lat.config(text=f"RTX Tensor Latence:   {self.engine.gpu_time_us:5.1f} µs")
        self.t_mops.config(text=f"AI Throughput:        {mops:5.2f} Mops/s")
        self.t_vram.config(text=f"VRAM Zero-Copy State: {vram_mb:5.1f} MB")

        # Aktualizace počtů živých pro každou frakci
        alive_per_faction = []
        for f in range(4):
            if f < self.cur_factions:
                f_alive = int(((self.engine.teams == f) & (self.engine.hp > 0)).sum())
                f_total = int((self.engine.teams == f).sum())
                self.faction_labels[f].config(text=f"• {FACTIONS[f]['name']}: {f_alive}/{f_total} živých", fg=FACTIONS[f]["hex"])
                alive_per_faction.append(f_alive)
            else:
                self.faction_labels[f].config(text=f"• [ NEAKTIVNÍ ]", fg="#3a3f55")

        self._render_frame()

        if self.frame_idx % 30 == 0:
            self.log_txt.insert(tk.END, f">> T#{self.frame_idx:04d} | Lasery: {len(self.engine.beams)} | GPU Lat: {self.engine.gpu_time_us:.1f}µs\n")
            self.log_txt.see(tk.END)

        # Kontrola vítěze (zůstala jen 1 frakce s živými)
        alive_factions_count = sum(1 for a in alive_per_faction if a > 0)
        if alive_factions_count <= 1 and self.frame_idx > 10:
            winner_idx = next((i for i, a in enumerate(alive_per_faction) if a > 0), -1)
            winner_name = FACTIONS[winner_idx]["name"] if winner_idx >= 0 else "NIKDO (VŠICHNI PADLI)"
            self.running = False
            self.btn_toggle.config(text="▶ ENGAGE WAR", bg="#00b4d8")
            self.log_txt.insert(tk.END, f"\n🏆 BITVA SKONČILA! Vítěz: {winner_name}!\n")
            self.log_txt.see(tk.END)
            return

        self.root.after(8, self._loop)

    def _render_frame(self):
        pos_np = self.engine.pos.cpu().numpy()
        hp_np = self.engine.hp.cpu().numpy()
        team_np = self.engine.teams.cpu().numpy()

        img = Image.new("RGBA", (self.CANVAS_W, self.CANVAS_H), (5, 6, 9, 255))
        draw = ImageDraw.Draw(img)

        # Cyber mřížka
        step = 50
        for x in range(0, self.CANVAS_W, step):
            draw.line([(x, 0), (x, self.CANVAS_H)], fill=(14, 17, 24, 255))
        for y in range(0, self.CANVAS_H, step):
            draw.line([(0, y), (self.CANVAS_W, y)], fill=(14, 17, 24, 255))

        # Středový radarový kruh
        cx, cy = self.CANVAS_W / 2, self.CANVAS_H / 2
        draw.ellipse([cx - 180, cy - 180, cx + 180, cy + 180], outline=(20, 26, 38, 255), width=1)
        draw.ellipse([cx - 70, cy - 70, cx + 70, cy + 70], outline=(25, 32, 48, 255), width=1)

        # Lasery
        for x1, y1, x2, y2, t in self.engine.beams:
            f_color = FACTIONS[t % 4]["color"]
            draw.line([(x1, y1), (x2, y2)], fill=f_color, width=1)

        # Výbuchy
        for x, y, age in self.engine.explosions:
            rad = age * 3.8
            alpha = max(0, 255 - age * 40)
            draw.ellipse([x - rad, y - rad, x + rad, y + rad], outline=(255, 183, 3, alpha), width=1)

        # Léčení
        for x, y, r in self.engine.heals:
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(56, 176, 0, 180), width=1)

        # Vykreslení bojovníků
        alive_mask = hp_np > 0
        r = 2.0 if self.cur_units >= 2000 else 3.0

        # Padlí (Vraky)
        dead_indices = np.where(~alive_mask)[0]
        for idx in dead_indices:
            x, y = pos_np[idx]
            draw.point((x, y), fill=(30, 34, 45, 160))

        # Živí boti s neonovou barvou své frakce
        alive_indices = np.where(alive_mask)[0]
        for idx in alive_indices:
            x, y = pos_np[idx]
            t = team_np[idx]
            draw.ellipse([x - r, y - r, x + r, y + r], fill=FACTIONS[t]["color"])

        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

def main():
    root = tk.Tk()
    app = CyberWarApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
