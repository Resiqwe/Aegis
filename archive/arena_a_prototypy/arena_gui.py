"""
arena_gui.py - Vizuální GUI aréna s telemetrií a logováním
Řízená Tenzorovým Křemíkem (Tensor AI Blob) na NVIDIA RTX 5070 Ti.

Optimalizováno pro stovky až tisíce botů:
  - Vektorizovaný výpočet prostorových vzdáleností (GPU / NumPy)
  - Možnost dynamicky měnit počet botů: 40, 100, 250, 500, 1000 jednotek
  - Telemetrie latence RTX 5070 Ti v mikrosekundách
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
from brick import Brick

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SIM_LOG_PATH = LOG_DIR / "arena_simulation.jsonl"

def game_ai_rules(s0_enemy, s1_low_hp, s2_has_medkit, s3_has_ammo):
    attack = int(s0_enemy and s3_has_ammo and not s1_low_hp)
    flee = int(s0_enemy and (not s3_has_ammo or s1_low_hp))
    heal = int(s1_low_hp and s2_has_medkit)
    return (attack, flee, heal)

def get_trained_weights():
    brick = Brick.learn("ARENA_BLOB", 4, 3, game_ai_rules, hidden=[8], max_epochs=4000, lr=0.08)
    return brick.params

class ArenaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎮 Tensor AI Blob - Masivní Herní Aréna (RTX 5070 Ti)")
        self.root.geometry("1180x760")
        self.root.configure(bg="#1a1a24")

        self.CANVAS_W = 760
        self.CANVAS_H = 640
        self.num_units = 200 # Výchozí počet botů (100 vs 100)

        self.running = False
        self.sim_speed_ms = 30
        self.frame_idx = 0
        self.total_decisions = 0

        # Příprava AI na GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        params = get_trained_weights()
        self.W1 = torch.tensor(params[0][0], dtype=torch.float32, device=self.device)
        self.b1 = torch.tensor(params[0][1], dtype=torch.float32, device=self.device)
        self.W2 = torch.tensor(params[1][0], dtype=torch.float32, device=self.device)
        self.b2 = torch.tensor(params[1][1], dtype=torch.float32, device=self.device)

        self.log_file = open(SIM_LOG_PATH, "w", encoding="utf-8")
        self._write_file_log({
            "event": "SIMULATION_INIT",
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "initial_units": self.num_units
        })

        self._build_ui()
        self.reset_simulation()

    def _build_ui(self):
        top_bar = tk.Frame(self.root, bg="#24263a", height=45)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        title_lbl = tk.Label(top_bar, text="🧠 TENSOR BLOB MASSIVE ARENA // RTX 5070 Ti", 
                             fg="#00f5d4", bg="#24263a", font=("Consolas", 12, "bold"))
        title_lbl.pack(side=tk.LEFT, padx=15, pady=8)

        self.dev_lbl = tk.Label(top_bar, text=f"HW: {torch.cuda.get_device_name(0)}", 
                                fg="#e0e1dd", bg="#24263a", font=("Consolas", 9))
        self.dev_lbl.pack(side=tk.RIGHT, padx=15)

        main_frame = tk.Frame(self.root, bg="#1a1a24")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Herní plátno
        left_box = tk.Frame(main_frame, bg="#0d0d12", bd=2, relief=tk.SOLID)
        left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left_box, width=self.CANVAS_W, height=self.CANVAS_H, bg="#0d0d12", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Postranní panel
        right_panel = tk.Frame(main_frame, width=380, bg="#1a1a24")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        # Ovládací panel
        btn_frame = tk.LabelFrame(right_panel, text=" OVLÁDÁNÍ SIMULACE ", bg="#1a1a24", fg="#00f5d4", font=("Consolas", 9, "bold"))
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        row1 = tk.Frame(btn_frame, bg="#1a1a24")
        row1.pack(fill=tk.X, padx=5, pady=5)

        self.btn_toggle = tk.Button(row1, text="▶ START", bg="#00b4d8", fg="#03045e", font=("Consolas", 10, "bold"),
                                    command=self.toggle_simulation, width=10, relief=tk.FLAT)
        self.btn_toggle.pack(side=tk.LEFT, padx=4)

        btn_reset = tk.Button(row1, text="↺ RESET", bg="#ef233c", fg="#ffffff", font=("Consolas", 10, "bold"),
                              command=self.reset_simulation, width=10, relief=tk.FLAT)
        btn_reset.pack(side=tk.LEFT, padx=4)

        # Volba počtu botů
        row2 = tk.Frame(btn_frame, bg="#1a1a24")
        row2.pack(fill=tk.X, padx=5, pady=5)

        units_lbl = tk.Label(row2, text="Počet botů:", bg="#1a1a24", fg="#e0e1dd", font=("Consolas", 8))
        units_lbl.pack(side=tk.LEFT, padx=4)

        self.unit_combo = ttk.Combobox(row2, values=["40", "100", "200", "400", "800"], width=6, state="readonly")
        self.unit_combo.set(str(self.num_units))
        self.unit_combo.bind("<<ComboboxSelected>>", self._on_unit_change)
        self.unit_combo.pack(side=tk.LEFT, padx=4)

        speed_lbl = tk.Label(row2, text="Pauza (ms):", bg="#1a1a24", fg="#e0e1dd", font=("Consolas", 8))
        speed_lbl.pack(side=tk.LEFT, padx=(8, 2))
        self.speed_slider = tk.Scale(row2, from_=5, to=100, orient=tk.HORIZONTAL, bg="#1a1a24", fg="#00f5d4",
                                     highlightthickness=0, length=70, showvalue=False, command=self._on_speed_change)
        self.speed_slider.set(self.sim_speed_ms)
        self.speed_slider.pack(side=tk.LEFT, padx=2)

        # Telemetrický box
        telem_frame = tk.LabelFrame(right_panel, text=" TELEMETRIE V REÁLNÉM ČASE ", bg="#1a1a24", fg="#00f5d4", font=("Consolas", 9, "bold"))
        telem_frame.pack(fill=tk.X, pady=(0, 10))

        self.lbl_fps = tk.Label(telem_frame, text="Framerate:          -- FPS", bg="#1a1a24", fg="#ffffff", font=("Consolas", 9), anchor="w")
        self.lbl_fps.pack(fill=tk.X, padx=10, pady=2)

        self.lbl_gpu_time = tk.Label(telem_frame, text="RTX Tensor Latence:    -- µs", bg="#1a1a24", fg="#f72585", font=("Consolas", 9, "bold"), anchor="w")
        self.lbl_gpu_time.pack(fill=tk.X, padx=10, pady=2)

        self.lbl_team_a = tk.Label(telem_frame, text="Tým A (Modří):      -- živých", bg="#1a1a24", fg="#48cae4", font=("Consolas", 9), anchor="w")
        self.lbl_team_a.pack(fill=tk.X, padx=10, pady=2)

        self.lbl_team_b = tk.Label(telem_frame, text="Tým B (Červení):    -- živých", bg="#1a1a24", fg="#ff4d6d", font=("Consolas", 9), anchor="w")
        self.lbl_team_b.pack(fill=tk.X, padx=10, pady=2)

        self.lbl_decisions = tk.Label(telem_frame, text="Rozhodnutí celkem:  0", bg="#1a1a24", fg="#a2d2ff", font=("Consolas", 9), anchor="w")
        self.lbl_decisions.pack(fill=tk.X, padx=10, pady=(2, 6))

        # Logovací okno
        log_frame = tk.LabelFrame(right_panel, text=" ŽIVÝ AUDITNÍ LOG ", bg="#1a1a24", fg="#00f5d4", font=("Consolas", 9, "bold"))
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, bg="#0d0d12", fg="#a7c957", font=("Consolas", 8), height=14, bd=0)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        scroll = tk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scroll.set)

    def _on_unit_change(self, event):
        new_val = int(self.unit_combo.get())
        if new_val != self.num_units:
            self.num_units = new_val
            self.reset_simulation()

    def _on_speed_change(self, val):
        self.sim_speed_ms = int(val)

    def _write_file_log(self, data: dict):
        data["timestamp_ms"] = int(time.time() * 1000)
        data["frame"] = self.frame_idx
        self.log_file.write(json.dumps(data) + "\n")
        self.log_file.flush()

    def _log_ui(self, msg: str):
        self.log_text.insert(tk.END, f"[{self.frame_idx:04d}] {msg}\n")
        self.log_text.see(tk.END)

    def reset_simulation(self):
        self.running = False
        self.btn_toggle.config(text="▶ START", bg="#00b4d8")
        self.frame_idx = 0
        self.total_decisions = 0

        # Vektorizované pole pro bleskovou simulaci stovek botů:
        # pos: (N, 2), team: (N,), hp: (N,), ammo: (N,), medkits: (N,)
        N = self.num_units
        self.pos = np.zeros((N, 2), dtype=np.float32)
        self.team = np.array([i % 2 for i in range(N)], dtype=np.int8)
        self.hp = np.full(N, 100, dtype=np.int16)
        self.ammo = np.random.randint(3, 9, size=N, dtype=np.int16)
        self.medkits = np.random.choice([0, 1, 1], size=N).astype(np.int8)
        self.beams = [] # Střely pro vykreslení [(x1,y1, x2,y2, team)]
        self.heals = [] # Pozice léčení [(x,y)]

        # Rozmístění: Tým 0 vlevo, Tým 1 vpravo
        margin = 30
        for i in range(N):
            if self.team[i] == 0:
                self.pos[i, 0] = random.uniform(margin, self.CANVAS_W * 0.4)
                self.pos[i, 1] = random.uniform(margin, self.CANVAS_H - margin)
            else:
                self.pos[i, 0] = random.uniform(self.CANVAS_W * 0.6, self.CANVAS_W - margin)
                self.pos[i, 1] = random.uniform(margin, self.CANVAS_H - margin)

        self.log_text.delete("1.0", tk.END)
        self._log_ui(f"Simulace připravena: {self.num_units} botů ({self.num_units//2} vs {self.num_units//2}).")
        self._write_file_log({"event": "RESET", "num_units": self.num_units})
        self._redraw_canvas()

    def toggle_simulation(self):
        self.running = not self.running
        if self.running:
            self.btn_toggle.config(text="⏸ PAUZA", bg="#ffd166")
            self._log_ui("Simulace SPUŠTĚNA.")
            self._write_file_log({"event": "RESUME"})
            self.root.after(self.sim_speed_ms, self._step_simulation)
        else:
            self.btn_toggle.config(text="▶ POKRAČOVAT", bg="#00b4d8")
            self._log_ui("Simulace POZASTAVENA.")
            self._write_file_log({"event": "PAUSE"})

    def _step_simulation(self):
        if not self.running:
            return

        t_loop_start = time.perf_counter()
        self.frame_idx += 1
        N = self.num_units
        alive_mask = self.hp > 0
        alive_indices = np.where(alive_mask)[0]

        self.beams.clear()
        self.heals.clear()

        # -------------------------------------------------------------
        # 1. VEKTORIZOVANÝ SBĚR SENZORŮ PRO VŠECHNY BOTY
        # -------------------------------------------------------------
        # Vzdálenosti mezi všemi živými jednotkami
        sensors = np.zeros((N, 4), dtype=np.float32)

        if len(alive_indices) > 0:
            alive_pos = self.pos[alive_indices]
            alive_teams = self.team[alive_indices]

            # Rychlá matice vzdáleností
            diff = alive_pos[:, np.newaxis, :] - alive_pos[np.newaxis, :, :]
            dist_matrix = np.hypot(diff[:, :, 0], diff[:, :, 1])
            
            # Maskování vlastního týmu
            enemy_mask = alive_teams[:, np.newaxis] != alive_teams[np.newaxis, :]
            dist_matrix[~enemy_mask] = 99999.0

            # Pro každého živého najdeme nejbližšího nepřítele
            nearest_enemy_dist = np.min(dist_matrix, axis=1)
            nearest_enemy_idx_local = np.argmin(dist_matrix, axis=1)
            nearest_enemy_global = alive_indices[nearest_enemy_idx_local]

            # Plnění senzorů
            # S0: nepřítel do 120 px
            sensors[alive_indices, 0] = (nearest_enemy_dist <= 120.0).astype(np.float32)
            # S1: HP pod 35%
            sensors[alive_indices, 1] = (self.hp[alive_indices] <= 35).astype(np.float32)
            # S2: Má lékárničku
            sensors[alive_indices, 2] = (self.medkits[alive_indices] > 0).astype(np.float32)
            # S3: Má munici
            sensors[alive_indices, 3] = (self.ammo[alive_indices] > 0).astype(np.float32)

        # -------------------------------------------------------------
        # 2. INFERENCE NA RTX 5070 Ti (CUDA) - VŠECHNY JEDNOTKY NARÁZ
        # -------------------------------------------------------------
        t_gpu_0 = time.perf_counter()
        sensors_gpu = torch.from_numpy(sensors).to(self.device)
        h = torch.tanh(sensors_gpu @ self.W1 + self.b1)
        actions_gpu = (torch.sigmoid(h @ self.W2 + self.b2) >= 0.5).to(torch.int8)
        torch.cuda.synchronize()
        gpu_time_us = (time.perf_counter() - t_gpu_0) * 1_000_000

        decisions = actions_gpu.cpu().numpy()
        self.total_decisions += N

        # -------------------------------------------------------------
        # 3. PROVEDENÍ AKCÍ
        # -------------------------------------------------------------
        actions_count = {"ATK": 0, "FLEE": 0, "HEAL": 0, "PATROL": 0}

        for idx_in_alive, u_idx in enumerate(alive_indices):
            atk, flee, heal = decisions[u_idx]
            target_enemy = nearest_enemy_global[idx_in_alive]
            has_valid_enemy = nearest_enemy_dist[idx_in_alive] < 90000.0

            if heal:
                self.hp[u_idx] = min(100, self.hp[u_idx] + 40)
                self.medkits[u_idx] -= 1
                self.heals.append((self.pos[u_idx, 0], self.pos[u_idx, 1]))
                actions_count["HEAL"] += 1
            elif atk and has_valid_enemy:
                self.ammo[u_idx] -= 1
                dmg = random.randint(18, 30)
                self.hp[target_enemy] -= dmg
                self.beams.append((self.pos[u_idx, 0], self.pos[u_idx, 1], 
                                  self.pos[target_enemy, 0], self.pos[target_enemy, 1], self.team[u_idx]))
                actions_count["ATK"] += 1
            elif flee and has_valid_enemy:
                dx = self.pos[u_idx, 0] - self.pos[target_enemy, 0]
                dy = self.pos[u_idx, 1] - self.pos[target_enemy, 1]
                norm = max(1.0, np.hypot(dx, dy))
                self.pos[u_idx, 0] = np.clip(self.pos[u_idx, 0] + (dx / norm) * 4.0, 15, self.CANVAS_W - 15)
                self.pos[u_idx, 1] = np.clip(self.pos[u_idx, 1] + (dy / norm) * 4.0, 15, self.CANVAS_H - 15)
                actions_count["FLEE"] += 1
            else:
                # Pohyb směrem k nepříteli
                if has_valid_enemy:
                    dx = self.pos[target_enemy, 0] - self.pos[u_idx, 0]
                    dy = self.pos[target_enemy, 1] - self.pos[u_idx, 1]
                    norm = max(1.0, np.hypot(dx, dy))
                    self.pos[u_idx, 0] = np.clip(self.pos[u_idx, 0] + (dx / norm) * 2.5, 15, self.CANVAS_W - 15)
                    self.pos[u_idx, 1] = np.clip(self.pos[u_idx, 1] + (dy / norm) * 2.5, 15, self.CANVAS_H - 15)
                actions_count["PATROL"] += 1

        # -------------------------------------------------------------
        # 4. TELEMETRIE A LOG
        # -------------------------------------------------------------
        dt_total = time.perf_counter() - t_loop_start
        fps = 1.0 / dt_total if dt_total > 0 else 0

        team_a_alive = int(np.sum((self.team == 0) & (self.hp > 0)))
        team_b_alive = int(np.sum((self.team == 1) & (self.hp > 0)))
        half = N // 2

        self.lbl_fps.config(text=f"Framerate:          {fps:5.1f} FPS")
        self.lbl_gpu_time.config(text=f"RTX Tensor Latence:    {gpu_time_us:5.1f} µs")
        self.lbl_team_a.config(text=f"Tým A (Modří):      {team_a_alive:3d} / {half} živých")
        self.lbl_team_b.config(text=f"Tým B (Červení):    {team_b_alive:3d} / {half} živých")
        self.lbl_decisions.config(text=f"Rozhodnutí celkem:  {self.total_decisions:,}")

        if self.frame_idx % 15 == 0 or actions_count["ATK"] > 0:
            self._write_file_log({
                "team_a": team_a_alive,
                "team_b": team_b_alive,
                "gpu_lat_us": round(gpu_time_us, 2),
                "fps": round(fps, 1),
                "actions": actions_count
            })

        self._redraw_canvas()

        if team_a_alive == 0 or team_b_alive == 0:
            winner = "TÝM A (Modří)" if team_a_alive > 0 else "TÝM B (Červení)"
            self.running = False
            self.btn_toggle.config(text="▶ START", bg="#00b4d8")
            self._log_ui(f"🏆 VÍTĚZSTVÍ: {winner} vyhrál bitvu!")
            self._write_file_log({"event": "GAME_OVER", "winner": winner})
            return

        self.root.after(self.sim_speed_ms, self._step_simulation)

    def _redraw_canvas(self):
        self.canvas.delete("all")

        # Paprsky střelby
        for x1, y1, x2, y2, t in self.beams:
            color = "#00f5d4" if t == 0 else "#ff0054"
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=1.5)

        # Pulzy léčení
        for x, y in self.heals:
            self.canvas.create_oval(x - 12, y - 12, x + 12, y + 12, outline="#38b000", width=1.5)

        # Kreslení botů podle počtu (menší poloměr pro víc jednotek)
        r = 5 if self.num_units <= 100 else (4 if self.num_units <= 400 else 2.5)

        for i in range(self.num_units):
            x, y = self.pos[i, 0], self.pos[i, 1]
            if self.hp[i] <= 0:
                # Malý padlý křížek
                self.canvas.create_line(x - 2, y - 2, x + 2, y + 2, fill="#2b2d42")
                continue

            color = "#00b4d8" if self.team[i] == 0 else "#e63946"
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="")

            # Health-bar jen pokud není botů příliš mnoho
            if self.num_units <= 200:
                bar_w = 12
                hp_ratio = max(0.0, self.hp[i] / 100.0)
                bx1 = x - bar_w / 2
                by1 = y - r - 4
                self.canvas.create_rectangle(bx1, by1, bx1 + bar_w, by1 + 2, fill="#2b2d42", outline="")
                self.canvas.create_rectangle(bx1, by1, bx1 + bar_w * hp_ratio, by1 + 2, 
                                            fill="#06d6a0" if hp_ratio > 0.4 else "#e63946", outline="")

    def on_closing(self):
        self.running = False
        try:
            self.log_file.close()
        except:
            pass
        self.root.destroy()

def main():
    root = tk.Tk()
    app = ArenaGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
