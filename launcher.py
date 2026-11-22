"""
launcher.py - Cyber-Tactics // Velitelské Centrum & AI Simulátor (Next-Gen 180Hz Edition)
Vlastnosti:
  1. Nativní podpora vysokých obnovovacích frekvencí (144Hz / 180Hz / 240Hz / Bez omezení).
  2. Elegantní Modern Dark design: čistá typografie Segoe UI / Bahnschrift, zaoblené panely, žádné černé díry.
  3. Text Cache & Backbuffer: nulový input lag, blesková odezva myši.
  4. Jednookenní architektura se sdíleným PyTorch CUDA kontextem a Hot-Reloadem.
"""

import os
import sys

# ── Project root detection & sys.path setup ───────────────────────
# Zajištění přístupu k rootu projektu (test) i k podsložce (src)
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, "src")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if os.path.isdir(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Ochrana pro běh pod pythonw.exe
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

# Self-bootstrap do .venv
try:
    import torch
    import pygame
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for venv_py in [
        os.path.join(script_dir, ".venv", "Scripts", "python.exe"),
        os.path.join(script_dir, "..", ".venv", "Scripts", "python.exe"),
    ]:
        if os.path.exists(venv_py) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_py).lower():
            import subprocess
            sys.exit(subprocess.call([venv_py] + sys.argv))

import glob
import json
import time
import importlib
import webbrowser

import numpy as np
import torch
import pygame

import scenarios
import ai_brains
import engine_core
import telemetry_logger
import tactical_hud
import battlefield_renderer as bf_render
import analyze_telemetry
import telemetry_lab
import hardware_telemetry
from src.engine.state_bridge import LockFreeStateBridge
from src.engine.async_runner import AsyncSimulationWorker
from src.ui.widgets import (
    BG_VOID, PANEL_SURFACE, CARD_BG, BORDER_SUBTLE, TEXT_BODY, TEXT_MUTED,
    ACCENT_CYAN, ACCENT_AMBER, ACCENT_GREEN, ACCENT_ROSE, FACTION_THEMES, CLASS_COLORS,
    ModernTextCache, draw_card as _draw_card, draw_launcher_button as _draw_launcher_button,
)
from src.ui.launcher_panels import (
    draw_modern_dock as _draw_modern_dock,
    draw_modern_radar as _draw_modern_radar,
    draw_graphical_heatmap as _draw_graphical_heatmap,
    draw_modern_analytics as _draw_modern_analytics,
)


class CyberTacticsLauncher:
    def __init__(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        pygame.init()
        pygame.display.set_caption("CYBER-TACTICS // VELITELSKÝ DOK & AI SIMULÁTOR")

        info = pygame.display.Info()
        disp_w = info.current_w if info.current_w > 0 else 1920
        disp_h = info.current_h if info.current_h > 0 else 1080

        # Automatické přizpůsobení monitoru (UWQHD 3440x1440 nebo QHD)
        if disp_w >= 3000:
            self.W = int(disp_w * 0.94)
            self.H = int(disp_h * 0.92)
        else:
            self.W = min(1760, max(1280, int(disp_w * 0.82)))
            self.H = min(1020, max(800, int(disp_h * 0.86)))

        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.text_cache = ModernTextCache()

        # Profil obnovovací frekvence: Podpora pro 180Hz monitory
        self.target_fps_presets = [60, 120, 180, 240, 0]
        self.target_fps_idx = 2  # Výchozí: 180 FPS
        self.measured_fps = 180.0

        self.has_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.has_cuda else "cpu")
        self.gpu_name = torch.cuda.get_device_name(0) if self.has_cuda else "CPU Fallback"
        self.vram_gb = (
            round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
            if self.has_cuda else 0.0
        )

        self.selected_scenario = 0
        self.budget_presets = [150, 250, 350, 500] # Body na frakci
        self.unit_presets = [b * 4 for b in self.budget_presets]
        self.selected_units_idx = 1

        self.telemetry_dir = "telemetry"
        os.makedirs(self.telemetry_dir, exist_ok=True)
        self.snapshots = []
        self.selected_snapshot_idx = 0
        self.snapshot_scroll_offset = 0
        self.epochs_data = []
        self.heatmap_grid = np.zeros((22, 36), dtype=np.int32)
        self.total_spatial_events = 0
        self.last_refresh_time = 0.0

        # Sdílený výpočetní engine, Lock-Free Bridge a Asynchronní Worker
        self.engine = engine_core.RealisticBattleEngine(1600, 1000, self.device)
        self.telemetry = telemetry_logger.TelemetryManager(self.telemetry_dir)
        self.engine.telemetry_mgr = self.telemetry
        self.engine.configure(self.unit_presets[self.selected_units_idx], 4)

        self.bridge = LockFreeStateBridge()
        self.worker = AsyncSimulationWorker(self.engine, self.bridge)
        self.last_known_state = None

        # Stav simulace
        self.sim_active = False
        self.sim_paused = False
        self.start_from_snapshot = False
        self.view_mode = "RADAR"  # "RADAR" nebo "HEATMAP"
        self.sim_speed_idx = 1  # 1x (reálný čas)
        self.speed_presets = ["0.5x", "1x", "2x", "4x", "8x", "MAX"]
        self.speed_mults = [0.5, 1.0, 2.0, 4.0, 8.0, 15.0]

        self.sim_step_counter = 0
        self.sim_tps_timer = time.perf_counter()
        self.sim_tps = 0.0
        self.sim_target_epochs = 0
        self.sim_done_epochs = 0
        self.sim_start_epoch = 1
        self.sim_last_epoch = 1
        self.sim_start_time = 0.0
        self._bg_acc = 0.0
        self._last_bg_time = time.perf_counter()
        self.sim_status_line = "Systém připraven. Výpočet i vizualizace sdílejí vysoce optimalizovaný pipeline."
        self.status_banner = "Vítej v Cyber-Tactics. Prozkoumej taktický radar nebo spusť kontinuální trénink."
        self.status_color = ACCENT_CYAN

        self.ui_buttons = {}
        self.snapshot_rects = []

        self.hw_tracker = hardware_telemetry.HardwareTelemetryTracker(self.telemetry_dir)
        self.hw_metrics = self.hw_tracker.current_metrics
        self.last_step_duration_ms = 0.0
        self.last_render_duration_ms = 0.0

        self._rebuild_fonts()
        self.refresh_telemetry_data()

    def _rebuild_fonts(self):
        w, h = self.screen.get_size()
        # Škálovací faktor přizpůsobený ultrawide / 1440p
        self.scale = max(1.0, min(1.65, min(w / 1920.0, h / 1080.0) * 1.25))
        s = self.scale

        # Čisté moderní bezpatkové písmo (Segoe UI -> Arial fallback)
        font_name = "segoeui"
        self.f_hero = pygame.font.SysFont(font_name, max(18, int(22 * s)), bold=True)
        self.f_kpi = pygame.font.SysFont(font_name, max(24, int(32 * s)), bold=True)
        self.f_title = pygame.font.SysFont(font_name, max(13, int(16 * s)), bold=True)
        self.f_sub = pygame.font.SysFont(font_name, max(11, int(13 * s)), bold=True)
        self.f_body = pygame.font.SysFont(font_name, max(11, int(12.5 * s)))
        self.f_small = pygame.font.SysFont(font_name, max(10, int(11.5 * s)))
        self.f_tiny = pygame.font.SysFont(font_name, max(9, int(10 * s)))

    def set_status(self, text: str, color=ACCENT_CYAN):
        self.status_banner = text
        self.status_color = color

    def hot_reload_modules(self):
        try:
            importlib.reload(scenarios)
            importlib.reload(ai_brains)
            importlib.reload(engine_core)
            importlib.reload(telemetry_logger)
            importlib.reload(tactical_hud)
            importlib.reload(bf_render)
            importlib.reload(analyze_telemetry)
            importlib.reload(telemetry_lab)
            importlib.reload(hardware_telemetry)
            return True
        except Exception as e:
            self.set_status(f"Chyba při Hot-Reloadu: {e}", ACCENT_ROSE)
            return False

    def refresh_telemetry_data(self):
        self.last_refresh_time = time.time()
        snaps = glob.glob(os.path.join(self.telemetry_dir, "snapshot_*.pt"))
        snaps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        self.snapshots = snaps
        if self.selected_snapshot_idx >= len(self.snapshots):
            self.selected_snapshot_idx = 0

        logs = glob.glob(os.path.join(self.telemetry_dir, "telemetry_session_*.jsonl"))
        logs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        target_log = None
        for lg in logs:
            if os.path.getsize(lg) > 1000:
                target_log = lg
                break

        self.epochs_data = []
        session_id = ""
        if target_log and os.path.exists(target_log):
            session_id = os.path.basename(target_log).replace("telemetry_session_", "").replace(".jsonl", "")
            try:
                with open(target_log, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rec = json.loads(line)
                            if rec.get("type") == "EPOCH_COMPLETED":
                                self.epochs_data.append(rec)
            except (OSError, json.JSONDecodeError):
                # Očekávané při poškozeném nebo rozepsaném JSONL logu
                pass

        self.heatmap_grid.fill(0)
        self.total_spatial_events = 0
        pq_files = glob.glob(os.path.join(self.telemetry_dir, "spatial_events_*.parquet"))
        pq_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        if pq_files:
            try:
                import pandas as pd
                df = pd.read_parquet(pq_files[0])
                self.total_spatial_events = len(df)
                if len(df) > 0:
                    from config import WORLD_W, WORLD_H
                    xs = np.clip((df["x"].to_numpy() / float(WORLD_W)) * 36, 0, 35).astype(np.int32)
                    ys = np.clip((df["y"].to_numpy() / float(WORLD_H)) * 22, 0, 21).astype(np.int32)
                    np.add.at(self.heatmap_grid, (ys, xs), 1)
            except (ImportError, OSError, KeyError, ValueError):
                # Očekávané pokud chybí pandas, parquet engine nebo soubor nebyl dočten
                pass

    def refresh_telemetry_live(self):
        self.last_refresh_time = time.time()
        if self.telemetry:
            self.epochs_data = list(self.telemetry.get_live_history())
            self.heatmap_grid = self.telemetry.build_heatmap_grid(36, 22, float(self.engine.W), float(self.engine.H))
            self.total_spatial_events = len(self.telemetry.spatial_events)

    def start_simulation(self, target_epochs: int = 0, use_snapshot: bool = False):
        if not self.hot_reload_modules():
            return
        if not self.sim_active:
            if use_snapshot:
                if not self.snapshots or not (0 <= self.selected_snapshot_idx < len(self.snapshots)):
                    self.set_status("Chyba: Zvolen režim ZE SNAPSHOTU, ale není vybrán žádný model!", ACCENT_ROSE)
                    return
                snap_path = self.snapshots[self.selected_snapshot_idx]
                if not self.telemetry.load_snapshot(self.engine, snap_path):
                    self.set_status(f"Chyba při načítání snapshotu: {os.path.basename(snap_path)}", ACCENT_ROSE)
                    return
                self.set_status(f"Načten model: {os.path.basename(snap_path)}", ACCENT_CYAN)
            else:
                n_u = self.unit_presets[self.selected_units_idx]
                self.engine.configure(n_u, 4)
                self.engine.switch_scenario(self.selected_scenario)

            self.sim_active = True
            self.sim_paused = False
            self.sim_target_epochs = target_epochs
            self.sim_done_epochs = 0
            self.sim_start_epoch = self.engine.current_epoch
            self.sim_last_epoch = self.engine.current_epoch
            self.sim_start_time = time.perf_counter()
            self.sim_status_line = "Běží kontinuální simulace kampaně..."

            mult = self.speed_mults[self.sim_speed_idx]
            self.worker.set_speed(mult)
            self.worker.set_paused(False)
            if not self.worker.is_alive():
                self.worker.start()

            self.set_status("Simulace spuštěna! Mezerník otevírá živý pohled bojiště.", ACCENT_GREEN)

    def toggle_sim_pause(self):
        if self.sim_active:
            self.sim_paused = not self.sim_paused
            if self.worker.is_alive():
                self.worker.set_paused(self.sim_paused)
            self.set_status("Simulace pozastavena." if self.sim_paused else "Simulace pokračuje.", ACCENT_AMBER if self.sim_paused else ACCENT_GREEN)

    def stop_simulation(self, save: bool = True):
        if not self.sim_active:
            return
        self.sim_active = False
        self.sim_paused = False
        if self.worker.is_alive():
            self.worker.stop()
        if save:
            snap_path = self.telemetry.save_snapshot(self.engine, custom_tag="user_stop")
            self.refresh_telemetry_data()
            self.set_status(f"Uložen stav: {os.path.basename(snap_path)}", ACCENT_GREEN)

    def step_simulation_background(self):
        if not self.sim_active or self.sim_paused or getattr(self, "_preview_open", False):
            return

        # Řízení rychlosti workeru podle nastavení launcheru
        mult = self.speed_mults[self.sim_speed_idx]
        self.worker.set_speed(mult)
        self.worker.set_paused(self.sim_paused)

        # Načteme interpolovaný stav z lock-free mostu (bez zátěže GPU a bez volání engine.step!)
        st = self.bridge.get_interpolated_state()
        if st is not None:
            self.last_known_state = st

        # Sledování epoch a TPS z běžícího workeru
        self.sim_tps = self.worker.current_tps
        if self.engine.current_epoch > self.sim_last_epoch:
            self.sim_done_epochs += 1
            self.sim_last_epoch = self.engine.current_epoch
            self.telemetry.flush_spatial_parquet()
            self.refresh_telemetry_live()

            if self.sim_target_epochs > 0 and self.sim_done_epochs >= self.sim_target_epochs:
                self.stop_simulation(save=True)

    def open_live_preview(self):
        if getattr(self, "_preview_open", False):
            return
        if not self.sim_active:
            self.start_simulation(target_epochs=0, use_snapshot=self.start_from_snapshot)
        if not self.hot_reload_modules():
            return
        self._preview_open = True
        try:
            hud_app = tactical_hud.OptimizedTacticalHUD(
                existing_screen=self.screen,
                existing_engine=self.engine,
                existing_telemetry=self.telemetry,
                existing_bridge=self.bridge,
                existing_worker=self.worker,
                embedded=True,
            )
            hud_app.running = not self.sim_paused
            hud_app.run()
            self.sim_paused = not hud_app.running
            if self.worker.is_alive():
                self.worker.set_paused(self.sim_paused)
            pygame.display.set_caption("CYBER-TACTICS // VELITELSKÝ DOK & AI SIMULÁTOR")
            self.W, self.H = self.screen.get_size()
            self._rebuild_fonts()
            self.refresh_telemetry_data()
            self.set_status("Návrat z bojiště. Simulace plynule pokračuje v doku.", ACCENT_GREEN)
        except Exception as e:
            self.set_status(f"Chyba při otevření bojiště: {e}", ACCENT_ROSE)
        finally:
            self._preview_open = False
            # Vyčištění fronty kliknutí, aby zběsilé kliky z HUDu nepropadly do Launcheru
            pygame.event.clear(pygame.MOUSEBUTTONUP)
            pygame.event.clear(pygame.MOUSEBUTTONDOWN)

    def draw_card(self, rect: pygame.Rect, bg=PANEL_SURFACE, border=BORDER_SUBTLE, radius=8):
        _draw_card(self.screen, rect, bg=bg, border=border, radius=radius)

    def draw_button(self, key: str, rect: pygame.Rect, title: str, subtitle: str = "",
                    active: bool = False, accent=ACCENT_CYAN, radius=6, disabled: bool = False):
        _draw_launcher_button(
            self.screen, self.ui_buttons, self.text_cache, self.f_sub, self.f_tiny,
            key=key, rect=rect, title=title, subtitle=subtitle,
            active=active, accent=accent, radius=radius, disabled=disabled,
        )

    def render(self):
        W, H = self.screen.get_size()
        self.screen.fill(BG_VOID)
        self.ui_buttons.clear()
        self.snapshot_rects.clear()

        # =====================================================================
        # 1. HORNÍ APEX LIŠTA (COMMAND HEADER)
        # =====================================================================
        top_h = max(54, int(58 * self.scale))
        pygame.draw.rect(self.screen, PANEL_SURFACE, (0, 0, W, top_h))
        pygame.draw.line(self.screen, BORDER_SUBTLE, (0, top_h - 1), (W, top_h - 1), 1)

        t_logo = self.text_cache.render(self.f_hero, "⚡ CYBER-TACTICS", ACCENT_CYAN)
        self.screen.blit(t_logo, (20, (top_h - t_logo.get_height()) // 2))

        sub_info = f"GPU: {self.gpu_name} ({self.vram_gb} GB VRAM) • 180Hz Ready Engine • In-Process Hot-Reload"
        t_sub = self.text_cache.render(self.f_small, sub_info, TEXT_MUTED)
        self.screen.blit(t_sub, (20 + t_logo.get_width() + 16, (top_h - t_sub.get_height()) // 2 + 1))

        # Pravá tlačítka v hlavičce: Přepínač FPS a Obnovení dat
        fps_lbl = f"{self.target_fps_presets[self.target_fps_idx]} Hz" if self.target_fps_presets[self.target_fps_idx] > 0 else "MAX Hz"
        self.draw_button("toggle_fps", pygame.Rect(W - 270, (top_h - 32) // 2, 110, 32), f"🎯 {fps_lbl}", accent=ACCENT_AMBER)
        self.draw_button("refresh", pygame.Rect(W - 150, (top_h - 32) // 2, 130, 32), "🔄 OBNOVIT", accent=ACCENT_CYAN)

        # =====================================================================
        # 2. FRAKČNÍ HERO KARTY (TOP KPI STRIP)
        # =====================================================================
        bridge_state = self.bridge.get_interpolated_state()
        if bridge_state is not None:
            self.last_known_state = bridge_state
        state = self.last_known_state
        if state is None:
            state = {}
        strip_y = top_h + 10
        card_h = max(90, int(100 * self.scale))
        card_gap = 12
        card_w = (W - 24 - 3 * card_gap) // 4

        alive_counts = state.get("alive_counts", [0, 0, 0, 0])
        alive_by_class = state.get("alive_by_class", [])
        generals_meta = state.get("generals", [])
        kills_f = state.get("kills_by_faction", [0, 0, 0, 0])
        deaths_f = state.get("deaths_by_faction", [0, 0, 0, 0])
        wins_f = state.get("epoch_wins_per_faction", getattr(self.engine, "epoch_wins_per_faction", [0, 0, 0, 0]))
        init_counts = state.get("initial_team_counts", getattr(self.engine, "initial_team_counts", []))
        f_techs = state.get("faction_techs", getattr(self.engine, "faction_active_techs", [[] for _ in range(4)]))

        for f in range(4):
            cx = 12 + f * (card_w + card_gap)
            c_rect = pygame.Rect(cx, strip_y, card_w, card_h)
            theme = FACTION_THEMES[f]
            f_col = theme["color"]

            self.draw_card(c_rect, bg=CARD_BG, border=BORDER_SUBTLE, radius=8)
            # Horní accent linka
            pygame.draw.rect(self.screen, f_col, (cx + 8, strip_y, card_w - 16, 3), border_radius=2)

            f_alive = alive_counts[f] if f < len(alive_counts) else 0
            init_c = max(1, init_counts[f] if f < len(init_counts) else 100)
            eff_pct = int((f_alive / init_c) * 100) if self.sim_active else 100

            gen = generals_meta[f] if f < len(generals_meta) else {}
            lvl = gen.get("level", 0)
            xp = int(gen.get("xp", 0))

            # Název frakce a trofeje
            t_fname = self.text_cache.render(self.f_sub, theme["name"], f_col)
            self.screen.blit(t_fname, (cx + 14, strip_y + 8))

            t_wins = self.text_cache.render(self.f_small, f"🏆 {wins_f[f]} W", ACCENT_AMBER)
            self.screen.blit(t_wins, (cx + card_w - t_wins.get_width() - 14, strip_y + 9))

            # Velké Big-KPI Procento Přežití
            pct_col = ACCENT_GREEN if eff_pct > 35 else (ACCENT_AMBER if eff_pct > 15 else ACCENT_ROSE)
            kpi_txt = f"{eff_pct}%" if self.sim_active else "100%"
            t_kpi = self.text_cache.render(self.f_kpi, kpi_txt, pct_col)
            self.screen.blit(t_kpi, (cx + 14, strip_y + int(24 * self.scale)))

            t_units = self.text_cache.render(self.f_small, f"{f_alive} / {init_c} jednotek", TEXT_BODY)
            self.screen.blit(t_units, (cx + 18 + t_kpi.get_width(), strip_y + int(32 * self.scale)))

            # Statistika K/D a Lvl
            k_val = kills_f[f] if f < len(kills_f) else 0
            d_val = deaths_f[f] if f < len(deaths_f) else 0
            # Power-Ups frakce
            tech_tags = []
            if f < len(f_techs):
                tag_map = {0: "NANIT", 1: "RADAR", 2: "ŠTÍT", 3: "OC", 4: "SMOKE"}
                for tid in f_techs[f]:
                    tech_tags.append(tag_map.get(tid, f"T{tid}"))
            tech_str = (" [" + "][".join(tech_tags) + "]") if tech_tags else " [BEZ TECH]"

            t_meta = self.text_cache.render(self.f_tiny, f"K:{k_val} D:{d_val} • L{lvl} • TECH:{tech_str}", ACCENT_AMBER if tech_tags else TEXT_MUTED)
            self.screen.blit(t_meta, (cx + 14, strip_y + int(58 * self.scale)))

            # Stacked bar složení tříd v terénu
            strata = alive_by_class[f] if f < len(alive_by_class) else [0, 0, 0, 0]
            tot_s = sum(strata)
            bar_y = strip_y + card_h - 10
            bar_w = card_w - 28
            pygame.draw.rect(self.screen, (25, 30, 42), (cx + 14, bar_y, bar_w, 4), border_radius=2)
            if tot_s > 0:
                sx = cx + 14
                for ci, cnt in enumerate(strata):
                    if cnt > 0:
                        sw = max(1, int((cnt / tot_s) * bar_w))
                        pygame.draw.rect(self.screen, CLASS_COLORS[ci], (sx, bar_y, sw, 4), border_radius=1)
                        sx += sw

        # =====================================================================
        # 3. HLAVNÍ TĚLO: LEVÝ ŘÍDÍCÍ DOK + STŘEDNÍ TAKTICKÝ RADAR + PRAVÁ ANALYTIKA
        # =====================================================================
        main_y = strip_y + card_h + 12
        foot_h = max(26, int(28 * self.scale))
        main_h = H - main_y - foot_h - 10

        # Proporce: Levý dok (320px), Střední Holo-Radar (56% zbylého), Pravá Analytika (44% zbylého)
        dock_w = max(310, int(330 * self.scale))
        rem_w = W - dock_w - 32
        radar_w = int(rem_w * 0.58)
        right_w = rem_w - radar_w - 12

        dock_rect = pygame.Rect(12, main_y, dock_w, main_h)
        radar_rect = pygame.Rect(12 + dock_w + 12, main_y, radar_w, main_h)
        right_rect = pygame.Rect(12 + dock_w + 12 + radar_w + 12, main_y, right_w, main_h)

        self._draw_modern_dock(dock_rect, state)
        self._draw_modern_radar(radar_rect, state)
        self._draw_modern_analytics(right_rect, state)

        # =====================================================================
        # 4. SPODNÍ STAVOVÝ PÁS (FOOTER)
        # =====================================================================
        bar_y = H - foot_h
        pygame.draw.rect(self.screen, PANEL_SURFACE, (0, bar_y, W, foot_h))
        pygame.draw.line(self.screen, BORDER_SUBTLE, (0, bar_y), (W, bar_y), 1)

        t_stat = self.text_cache.render(self.f_small, f">> {self.status_banner}", self.status_color)
        self.screen.blit(t_stat, (16, bar_y + (foot_h - t_stat.get_height()) // 2))

        # FPS / TPS telemetrie vpravo dole
        fps_curr = int(self.clock.get_fps())
        tps_curr = int(self.sim_tps) if self.sim_active else 0
        speed_lbl = self.speed_presets[self.sim_speed_idx]
        run_info = f"GUI: {fps_curr} FPS  •  Simulace: {tps_curr} TPS ({speed_lbl})  •  Krok: {self.hw_metrics['step_ms']:.1f}ms"
        t_perf = self.text_cache.render(self.f_small, run_info, ACCENT_GREEN if self.sim_active else TEXT_MUTED)
        self.screen.blit(t_perf, (W - t_perf.get_width() - 18, bar_y + (foot_h - t_perf.get_height()) // 2))

    def _draw_modern_dock(self, rect: pygame.Rect, state: dict | None = None):
        _draw_modern_dock(self, rect, state)

    def _draw_modern_radar(self, rect: pygame.Rect, state: dict):
        _draw_modern_radar(self, rect, state)

    def _draw_graphical_heatmap(self, r_box: pygame.Rect):
        _draw_graphical_heatmap(self, r_box)

    def _draw_modern_analytics(self, rect: pygame.Rect, state: dict):
        _draw_modern_analytics(self, rect, state)

    def _on_refresh(self):
        self.refresh_telemetry_data()
        self.set_status("Data byla obnovena z disku.", ACCENT_CYAN)

    def _on_toggle_fps(self):
        self.target_fps_idx = (self.target_fps_idx + 1) % len(self.target_fps_presets)
        target = self.target_fps_presets[self.target_fps_idx]
        self.set_status(f"Cílová frekvence GUI přepnuta na {target if target > 0 else 'Bez omezení'} Hz.", ACCENT_AMBER)

    def _on_mode_new(self):
        self.start_from_snapshot = False
        self.set_status("Vybrán režim: Čistá kampaň (nová evoluce).", ACCENT_GREEN)

    def _on_mode_snap(self):
        if not self.snapshots:
            self.set_status("V archivu není žádný snapshot pro načtení!", ACCENT_ROSE)
        else:
            self.start_from_snapshot = True
            snap_n = os.path.basename(self.snapshots[self.selected_snapshot_idx])
            self.set_status(f"Vybrán režim: Ze snapshotu ({snap_n}).", ACCENT_CYAN)

    def _on_run_lab(self):
        lab = telemetry_lab.TelemetryLab(telemetry_dir=self.telemetry_dir)
        if lab.epochs_data:
            html = lab.generate_html_report()
            if html and os.path.exists(html):
                webbrowser.open(os.path.abspath(html))
            self.set_status("AI Report otevřen v prohlížeči.", ACCENT_GREEN)
        else:
            self.set_status("Zatím není dostatek dat pro report.", ACCENT_AMBER)

    def _on_select_scenario(self, scen_idx: int):
        if self.sim_active:
            self.set_status("Změna mapy není možná za běhu. Nejprve ukončete simulaci.", ACCENT_AMBER)
        else:
            self.selected_scenario = scen_idx
            self.engine.switch_scenario(self.selected_scenario)
            self.set_status(f"Nastavena mapa: {scenarios.SCENARIO_NAMES[self.selected_scenario].split('(')[0].strip()}", ACCENT_CYAN)

    def _on_select_units(self, u_idx: int):
        if self.sim_active:
            self.set_status("Změna počtu jednotek není možná za běhu. Nejprve ukončete simulaci.", ACCENT_AMBER)
        else:
            self.selected_units_idx = u_idx
            self.engine.configure(self.unit_presets[self.selected_units_idx], 4)
            self.set_status(f"Nastaven rozpočet operace: {self.budget_presets[self.selected_units_idx]} b. na frakci.", ACCENT_CYAN)

    def _dispatch_button(self, key: str):
        exact_handlers = {
            "refresh": self._on_refresh,
            "toggle_fps": self._on_toggle_fps,
            "mode_new": self._on_mode_new,
            "mode_snap": self._on_mode_snap,
            "start_continuous": lambda: self.start_simulation(target_epochs=0, use_snapshot=self.start_from_snapshot),
            "open_preview": self.open_live_preview,
            "toggle_pause": self.toggle_sim_pause,
            "stop_sim": lambda: self.stop_simulation(save=True),
            "tab_radar": lambda: setattr(self, "view_mode", "RADAR"),
            "tab_heatmap": lambda: setattr(self, "view_mode", "HEATMAP"),
            "export_csv": self._handle_export_csv,
            "run_lab": self._on_run_lab,
            "clean_logs": self._handle_clean_logs,
        }
        handler = exact_handlers.get(key)
        if handler is not None:
            handler()
            return
        if key.startswith("scen_"):
            self._on_select_scenario(int(key.split("_")[1]))
        elif key.startswith("units_"):
            self._on_select_units(int(key.split("_")[1]))
        elif key.startswith("start_batch_"):
            self.start_simulation(target_epochs=int(key.split("_")[2]), use_snapshot=self.start_from_snapshot)
        elif key.startswith("sim_speed_"):
            self.sim_speed_idx = int(key.split("_")[2])

    def handle_click(self, mx: int, my: int):
        try:
            for idx, r_rect in list(self.snapshot_rects):
                if r_rect.collidepoint(mx, my):
                    if 0 <= idx < len(self.snapshots):
                        self.selected_snapshot_idx = idx
                        snap_name = os.path.basename(self.snapshots[idx])
                        self.set_status(f"Vybrán model: {snap_name}", ACCENT_CYAN)
                    return

            for key, rect in list(self.ui_buttons.items()):
                if rect.collidepoint(mx, my):
                    self._dispatch_button(key)
                    return
        except Exception as e:
            self.set_status(f"Chyba kliknutí: {e}", ACCENT_ROSE)

    def _handle_clean_logs(self):
        """Bezpečně archivuje a promaže staré telemetrické logy."""
        if self.sim_active:
            self.set_status("Před smazáním archivu nejprve zastavte běžící simulaci!", ACCENT_ROSE)
            return
        archive_dir = os.path.join(self.telemetry_dir, "archiv_clean")
        os.makedirs(archive_dir, exist_ok=True)
        count = 0
        import shutil
        for pattern in ["*.jsonl", "*.parquet", "*.summary.json"]:
            for fp in glob.glob(os.path.join(self.telemetry_dir, pattern)):
                try:
                    dest = os.path.join(archive_dir, os.path.basename(fp))
                    if os.path.exists(dest):
                        os.remove(dest)
                    shutil.move(fp, dest)
                    count += 1
                except OSError:
                    # Očekávané pokud je soubor uzamčen běžícím procesem
                    pass
        self.epochs_data.clear()
        self.heatmap_grid.fill(0)
        self.total_spatial_events = 0
        self.refresh_telemetry_data()
        self.set_status(f"Archiv vyčištěn ({count} souborů přesunuto do telemetry/archiv_clean/).", ACCENT_GREEN)

    def _handle_export_csv(self):
        """Exportuje metriky epoch a prostorové události do reálných CSV souborů."""
        try:
            import pandas as pd
            self.telemetry.flush_spatial_parquet()
            exp_count = 0
            # 1. Export epoch
            if self.epochs_data:
                rows = []
                for ep in self.epochs_data:
                    row = {
                        "epoch": ep.get("epoch"),
                        "timestamp": ep.get("timestamp"),
                        "scenario": ep.get("scenario"),
                        "duration_sec": ep.get("duration_sec"),
                        "winner_team": ep.get("winner_team"),
                        "end_reason": ep.get("end_reason"),
                    }
                    for f in ep.get("factions", []):
                        fid = f.get("faction_id")
                        row[f"f{fid}_alive"] = f.get("alive_count")
                        row[f"f{fid}_kills"] = f.get("kills")
                        row[f"f{fid}_fitness_avg"] = f.get("fitness_avg")
                        row[f"f{fid}_level"] = f.get("level")
                    rows.append(row)
                df_ep = pd.DataFrame(rows)
                ep_csv_path = os.path.join(self.telemetry_dir, "export_epochs.csv")
                df_ep.to_csv(ep_csv_path, index=False, encoding="utf-8")
                exp_count += 1

            # 2. Export spatial
            pq_files = glob.glob(os.path.join(self.telemetry_dir, "spatial_events_*.parquet"))
            if pq_files:
                pq_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                df_sp = pd.read_parquet(pq_files[0])
                sp_csv_path = os.path.join(self.telemetry_dir, "export_spatial_events.csv")
                df_sp.to_csv(sp_csv_path, index=False, encoding="utf-8")
                exp_count += 1

            if exp_count > 0:
                self.set_status("Export hotov: vytvořen export_epochs.csv a export_spatial_events.csv.", ACCENT_GREEN)
            else:
                self.set_status("Zatím nejsou k dispozici žádná data pro CSV export.", ACCENT_AMBER)
        except Exception as e:
            self.set_status(f"Chyba při exportu CSV: {e}", ACCENT_ROSE)

    def run(self):
        running = True
        while running:
            t0_step = time.perf_counter()
            self.step_simulation_background()
            self.last_step_duration_ms = (time.perf_counter() - t0_step) * 1000.0

            if not self.sim_active and time.time() - self.last_refresh_time > 3.0:
                self.refresh_telemetry_data()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    if self.sim_active:
                        self.stop_simulation(save=True)
                    running = False
                elif ev.type == pygame.VIDEORESIZE:
                    self.W = max(1200, ev.w)
                    self.H = max(760, ev.h)
                    self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE | pygame.DOUBLEBUF)
                    self._rebuild_fonts()
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        if self.sim_active:
                            self.stop_simulation(save=True)
                        else:
                            running = False
                    elif ev.key == pygame.K_SPACE:
                        self.open_live_preview()
                    elif ev.key == pygame.K_RETURN:
                        if self.sim_active:
                            self.open_live_preview()
                        else:
                            self.start_simulation(target_epochs=0, use_snapshot=self.start_from_snapshot)
                    elif ev.key == pygame.K_p:
                        self.toggle_sim_pause()
                elif ev.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    if hasattr(self, "snapshot_box_rect") and self.snapshot_box_rect.collidepoint(mx, my):
                        self.snapshot_scroll_offset = max(0, self.snapshot_scroll_offset - ev.y)
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self.handle_click(*pygame.mouse.get_pos())

            t0_rend = time.perf_counter()
            self.render()
            pygame.display.flip()
            self.last_render_duration_ms = (time.perf_counter() - t0_rend) * 1000.0

            # Vzorkování HW telemetrie
            self.hw_metrics = self.hw_tracker.sample(
                step_ms=self.last_step_duration_ms,
                render_ms=self.last_render_duration_ms,
                ui_fps=self.clock.get_fps(),
                sim_tps=self.sim_tps,
            )

            target_hz = self.target_fps_presets[self.target_fps_idx]
            self.clock.tick(target_hz)

        self.hw_tracker.flush()
        pygame.quit()


if __name__ == "__main__":
    app = CyberTacticsLauncher()
    app.run()
