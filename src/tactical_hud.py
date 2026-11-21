"""
tactical_hud.py - Výukový Simulátor AI & Taktické Velitelství (60 FPS SDL2)
Klíčové vlastnosti:
  1. Kompletní Rentgen Neuronové Sítě (14 Senzorů -> 28 Skrytých Neuronů -> 4 Akce):
     - Zobrazuje všech 14 vstupních senzorů vojáka, matici aktivace 28 skrytých neuronů
       a 4 výstupní rozhodnutí v reálném čase.
  2. Živý Graf Učení AI (Fitness Evolution Curve) & ROI Preference Generálů:
     - Sleduje vývoj průměrné Fitness všech 4 frakcí napříč generacemi a naučené
       pravděpodobnosti nákupu tříd (Pěšák, Medik, Sniper, Juggernaut).
  3. Vizuální rozlišení tříd a bio-paprsky Medika:
     - Unikátní siluety pro Pěšáka, Medika (+kříž a zelené léčebné paprsky), Snipera a Juggernauta.
  4. Přesný Turbo Trénink (až 8x rychlost pomocí sub-steppingu bez ztráty přesnosti fyziky).
"""

import sys
import os

# Automatický self-bootstrap do virtuálního prostředí (.venv):
try:
    import torch
    import pygame
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, ".venv", "Scripts", "python.exe"),
        os.path.join(script_dir, "..", ".venv", "Scripts", "python.exe"),
    ]
    for venv_py in candidates:
        if os.path.exists(venv_py) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_py).lower():
            import subprocess
            res = subprocess.call([venv_py] + sys.argv)
            sys.exit(res)

import time
import math
import numpy as np
import torch
import pygame

from src.scenarios import SCENARIO_NAMES
from src.engine_core import RealisticBattleEngine, DOCTRINE_LIST
from src.ai_brains import CLASS_MEDIC
from src.telemetry_logger import TelemetryManager
from src.engine.state_bridge import LockFreeStateBridge
from src.engine.async_runner import AsyncSimulationWorker
import battlefield_renderer as bf_render
from src.profiler import render_prof
from src.ui.widgets import (
    FACTION_META, BG_DARK, PANEL_BG, PANEL_HEADER, BORDER_COL,
    ACCENT_CYAN, ACCENT_AMBER, ACCENT_GREEN, ACCENT_PINK, TEXT_MAIN, TEXT_DIM,
    safe_rgb, ModernTextCache, draw_hud_button as _draw_hud_button,
)
from src.ui.hud_inspector import draw_right_dock as _draw_right_dock


class OptimizedTacticalHUD:
    def __init__(
        self,
        existing_screen: pygame.Surface | None = None,
        existing_engine: RealisticBattleEngine | None = None,
        existing_telemetry: TelemetryManager | None = None,
        existing_bridge: LockFreeStateBridge | None = None,
        existing_worker: AsyncSimulationWorker | None = None,
        embedded: bool = False,
    ):
        self.embedded = embedded
        self.text_cache = ModernTextCache()
        if not pygame.get_init():
            pygame.init()
        pygame.display.set_caption("[BLESK] CYBER-TACTICS // VÝUKOVÝ SIMULÁTOR AI & NEURO-EVOLUCE")

        if existing_screen is not None:
            self.screen = existing_screen
        else:
            info = pygame.display.Info()
            disp_w = info.current_w if info.current_w > 0 else 1920
            disp_h = info.current_h if info.current_h > 0 else 1080

            win_w = min(1780, max(1280, int(disp_w * 0.78)))
            win_h = min(1020, max(820, int(disp_h * 0.85)))

            self.screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.fullscreen = False

        self.faction_colors = [meta["color"] for meta in FACTION_META]
        self.ui_scale = 1.0
        self._rebuild_fonts()

        self.sim_time_ms = 1.0
        self.render_time_ms = 1.0
        self.measured_fps = 60.0
        self.vram_mb = 0.0

        self.SIM_W = 2400
        self.SIM_H = 1350
        self.unit_presets = [150, 250, 350, 500]  # Rozpočet bodů na frakci (Skirmish, Battalion, Legion, Armageddon)
        self.speed_presets = [0.5, 1.0, 2.0, 4.0, 8.0]
        self.speed_idx = 1
        self.running = True
        self.target_fps = 60  # Plynulých 60 FPS pro uvolnění CPU a GILu pro simulační worker
        self._sim_accumulator = 0.0
        self._last_sim_time = time.perf_counter()

        if existing_engine is not None:
            self.engine = existing_engine
            self.device = existing_engine.device
            self.SIM_W = existing_engine.W
            self.SIM_H = existing_engine.H
            self.n_factions = existing_engine.n_factions
            self.telemetry = existing_telemetry or getattr(existing_engine, "telemetry_mgr", None) or TelemetryManager("telemetry")
            self.engine.telemetry_mgr = self.telemetry
            n_tot = getattr(existing_engine, "n_units", 400)
            if n_tot in self.unit_presets:
                self.unit_preset_idx = self.unit_presets.index(n_tot)
            else:
                self.unit_preset_idx = 1
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.engine = RealisticBattleEngine(self.SIM_W, self.SIM_H, self.device)
            self.telemetry = TelemetryManager("telemetry")
            self.engine.telemetry_mgr = self.telemetry
            self.unit_preset_idx = 1
            self.n_factions = 4
            self.reboot_simulation()

        if existing_bridge is not None and existing_worker is not None:
            self.bridge = existing_bridge
            self.worker = existing_worker
            self._owns_worker = False
        else:
            self.bridge = LockFreeStateBridge()
            self.worker = AsyncSimulationWorker(self.engine, self.bridge)
            self.worker.start()
            self._owns_worker = True

        self.zoom = 1.0
        self.cam_x = self.SIM_W * 0.5
        self.cam_y = self.SIM_H * 0.5
        self.panning = False
        self.pan_start_mouse = (0, 0)
        self.pan_start_cam = (self.cam_x, self.cam_y)

        self._set_selected_unit(-1)  # -1 = auto-track nejlepšího vojáka generace
        self.follow_selected = False
        self.ui_buttons = {}
        self._btn_flash = {}  # {key: timestamp} pro okamžitou vizuální odezvu stisku
        self.viewport_rect = pygame.Rect(0, 0, 100, 100)
        self.status_banner = ""
        self.status_banner_timer = 0.0

    def _set_selected_unit(self, unit_idx: int):
        self.selected_unit = int(unit_idx)
        if hasattr(self, "bridge") and self.bridge is not None:
            self.bridge.set_selected_unit(self.selected_unit, auto_select_best=(self.selected_unit < 0))

    def _rebuild_fonts(self):
        """DPI/resolution scaling: na QHD/UWQHD zvětší písmo o ~35-50 %."""
        w, h = self.screen.get_size()
        self.ui_scale = max(1.0, min(1.5, min(w / 1600.0, h / 1000.0)))
        s = self.ui_scale
        self.f_title = pygame.font.SysFont("segoeui", max(12, int(14 * s)), bold=True)
        self.f_sub = pygame.font.SysFont("segoeui", max(11, int(12.5 * s)), bold=True)
        self.f_reg = pygame.font.SysFont("segoeui", max(10, int(11.5 * s)))
        self.f_small = pygame.font.SysFont("segoeui", max(10, int(11.5 * s)), bold=True)
        self.f_tiny = pygame.font.SysFont("segoeui", max(9, int(10 * s)), bold=True)
        self.f_mono = pygame.font.SysFont("consolas", max(9, int(10.5 * s)))

    def show_banner(self, text: str, duration: float = 2.5):
        self.status_banner = text
        self.status_banner_timer = duration

    def reboot_simulation(self):
        b_pts = self.unit_presets[self.unit_preset_idx]
        if hasattr(self, "worker") and self.worker is not None:
            self.worker.post_command("configure", b_pts * self.n_factions, self.n_factions)
        else:
            self.engine.configure(b_pts * self.n_factions, self.n_factions)
        self._set_selected_unit(-1)

    def reset_camera(self):
        self.zoom = 1.0
        self.cam_x = self.SIM_W * 0.5
        self.cam_y = self.SIM_H * 0.5
        self.follow_selected = False

    def world_to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        vr = self.viewport_rect
        if self.zoom <= 1.01:
            return vr.x + int((wx / self.SIM_W) * vr.width), vr.y + int((wy / self.SIM_H) * vr.height)

        sub_w = self.SIM_W / self.zoom
        sub_h = self.SIM_H / self.zoom
        x0 = max(0.0, min(self.SIM_W - sub_w, self.cam_x - sub_w * 0.5))
        y0 = max(0.0, min(self.SIM_H - sub_h, self.cam_y - sub_h * 0.5))
        return vr.x + int(((wx - x0) / sub_w) * vr.width), vr.y + int(((wy - y0) / sub_h) * vr.height)

    def screen_to_world(self, sx: int, sy: int) -> tuple[float, float]:
        vr = self.viewport_rect
        rel_x = max(0.0, min(1.0, (sx - vr.x) / max(1.0, float(vr.width))))
        rel_y = max(0.0, min(1.0, (sy - vr.y) / max(1.0, float(vr.height))))

        if self.zoom <= 1.01:
            return rel_x * self.SIM_W, rel_y * self.SIM_H

        sub_w = self.SIM_W / self.zoom
        sub_h = self.SIM_H / self.zoom
        x0 = max(0.0, min(self.SIM_W - sub_w, self.cam_x - sub_w * 0.5))
        y0 = max(0.0, min(self.SIM_H - sub_h, self.cam_y - sub_h * 0.5))
        return x0 + rel_x * sub_w, y0 + rel_y * sub_h

    def handle_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            elif ev.type == pygame.VIDEORESIZE and not self.fullscreen:
                w = max(1280, ev.w)
                h = max(780, ev.h)
                self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE | pygame.DOUBLEBUF)
                self._rebuild_fonts()
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return False
                elif ev.key == pygame.K_SPACE:
                    self.running = not self.running
                elif ev.key == pygame.K_F11:
                    self.fullscreen = not self.fullscreen
                    self.screen = (
                        pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF)
                        if self.fullscreen
                        else pygame.display.set_mode((1720, 960), pygame.RESIZABLE | pygame.DOUBLEBUF)
                    )
                    self._rebuild_fonts()
                elif ev.key == pygame.K_e:
                    if hasattr(self, "worker") and self.worker is not None:
                        self.worker.post_command("force_evolve")
                    else:
                        self.engine.finish_operation_and_evolve()
                    self.show_banner(f"Epocha #{self.engine.current_epoch:02d}: Požadavek na evoluci zařazen")
                elif ev.key == pygame.K_t:
                    self.speed_idx = (self.speed_idx + 1) % len(self.speed_presets)
                    self.show_banner(f"Rychlost tréninku: {self.speed_presets[self.speed_idx]:g}x")
                elif ev.key == pygame.K_F5 or (ev.key == pygame.K_s and (pygame.key.get_mods() & pygame.KMOD_CTRL)):
                    if hasattr(self, "worker") and self.worker is not None:
                        self.worker.post_command("save_snapshot")
                        self.show_banner("Ukládání snapshotu na pozadí...", 1.5)
                    else:
                        path = self.telemetry.save_snapshot(self.engine)
                        self.show_banner(f"Uložen snapshot [F5/Ctrl+S]: {os.path.basename(path)}")
                elif ev.key == pygame.K_l:
                    snaps = self.telemetry.list_available_snapshots()
                    if snaps and self.telemetry.load_snapshot(self.engine, snaps[0]):
                        self.show_banner(f"Načten snapshot: {os.path.basename(snaps[0])}")
                elif ev.key == pygame.K_f:
                    self.follow_selected = not self.follow_selected
                elif ev.key in (pygame.K_HOME, pygame.K_c):
                    self.reset_camera()
                elif ev.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    scen_target = ev.key - pygame.K_1
                    if hasattr(self, "worker") and self.worker is not None:
                        self.worker.post_command("switch_scenario", scen_target)
                    else:
                        self.engine.switch_scenario(scen_target)

            elif ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if self.viewport_rect.collidepoint(mx, my):
                    old_wx, old_wy = self.screen_to_world(mx, my)
                    self.zoom = min(4.5, self.zoom * 1.20) if ev.y > 0 else max(1.0, self.zoom / 1.20)
                    self.cam_x = 0.70 * self.cam_x + 0.30 * old_wx
                    self.cam_y = 0.70 * self.cam_y + 0.30 * old_wy

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if ev.button == 1:
                    if not self._check_ui_click(mx, my) and self.viewport_rect.collidepoint(mx, my):
                        wx, wy = self.screen_to_world(mx, my)
                        pos_np = None
                        if hasattr(self, "last_state") and self.last_state and "pos" in self.last_state:
                            pos_np = self.last_state["pos"]
                        elif hasattr(self, "bridge") and self.bridge is not None:
                            snap = self.bridge.get_interpolated_state()
                            if snap and "pos" in snap:
                                pos_np = snap["pos"]
                        if pos_np is not None and len(pos_np) > 0:
                            dists = np.hypot(pos_np[:, 0] - wx, pos_np[:, 1] - wy)
                            min_d = np.min(dists)
                            if min_d <= 40.0:
                                self._set_selected_unit(int(np.argmin(dists)))
                            else:
                                self._set_selected_unit(-1)
                        else:
                            self._set_selected_unit(-1)

                elif ev.button in (2, 3) and self.viewport_rect.collidepoint(mx, my):
                    self.panning = True
                    self.follow_selected = False
                    self.pan_start_mouse = (mx, my)
                    self.pan_start_cam = (self.cam_x, self.cam_y)

            elif ev.type == pygame.MOUSEBUTTONUP and ev.button in (2, 3):
                self.panning = False

            elif ev.type == pygame.MOUSEMOTION and self.panning and self.zoom > 1.01:
                mx, my = pygame.mouse.get_pos()
                dx = mx - self.pan_start_mouse[0]
                dy = my - self.pan_start_mouse[1]
                scale_x = (self.SIM_W / self.zoom) / max(1, self.viewport_rect.width)
                scale_y = (self.SIM_H / self.zoom) / max(1, self.viewport_rect.height)
                self.cam_x = max(0.0, min(self.SIM_W, self.pan_start_cam[0] - dx * scale_x))
                self.cam_y = max(0.0, min(self.SIM_H, self.pan_start_cam[1] - dy * scale_y))

        keys = pygame.key.get_pressed()
        pan_spd = 12.0 / max(1.0, self.zoom * 0.6)
        if keys[pygame.K_w]:
            self.cam_y = max(0.0, self.cam_y - pan_spd)
            self.follow_selected = False
        if keys[pygame.K_s]:
            self.cam_y = min(float(self.SIM_H), self.cam_y + pan_spd)
            self.follow_selected = False
        if keys[pygame.K_a]:
            self.cam_x = max(0.0, self.cam_x - pan_spd)
            self.follow_selected = False
        if keys[pygame.K_d]:
            self.cam_x = min(float(self.SIM_W), self.cam_x + pan_spd)
            self.follow_selected = False

        return True

    def _check_ui_click(self, mx: int, my: int) -> bool:
        for key, rect in self.ui_buttons.items():
            if rect.collidepoint(mx, my):
                self._trigger_button(key)
                return True
        return False

    def _on_next_epoch(self):
        if hasattr(self, "worker") and self.worker is not None:
            self.worker.post_command("force_evolve")
        else:
            self.engine.finish_operation_and_evolve()
        self.show_banner(f"Epocha #{self.engine.current_epoch:02d}: Požadavek na novou epochu odeslán", 3.0)

    def _on_cycle_scenario(self):
        next_scen = (self.engine.current_scenario_idx + 1) % 3
        if hasattr(self, "worker") and self.worker is not None:
            self.worker.post_command("switch_scenario", next_scen)
        else:
            self.engine.switch_scenario(next_scen)
        scen_name = SCENARIO_NAMES[next_scen].split('(')[0].strip()
        self.show_banner(f"[MAPA] Scénář přepnut: {scen_name}", 3.0)

    def _on_save_snap(self):
        if hasattr(self, "worker") and self.worker is not None:
            self.worker.post_command("save_snapshot")
            self.show_banner("Ukládání snapshotu na pozadí...", 1.5)
        else:
            path = self.telemetry.save_snapshot(self.engine)
            self.show_banner(f"[ULOZIT] Uložen model: {os.path.basename(path)}", 3.0)

    def _on_load_snap(self):
        snaps = self.telemetry.list_available_snapshots()
        if snaps and self.telemetry.load_snapshot(self.engine, snaps[0]):
            self.show_banner(f"[NACIST] Načten model: {os.path.basename(snaps[0])}", 3.0)
        else:
            self.show_banner("[VAROVANI] Žádný snapshot k načtení nenalezen!", 3.0)

    def _on_cycle_doctrine(self, f_id: int):
        new_doc = (self.engine.faction_doctrines[f_id] + 1) % 4
        if hasattr(self, "worker") and self.worker is not None:
            self.worker.post_command("set_doctrine", f_id, new_doc)
        else:
            self.engine.set_faction_doctrine(f_id, new_doc)
        doc_name = DOCTRINE_LIST[new_doc]
        doc_desc = {
            "BALANCED": "Vyvážený taktický postup a obsazování",
            "RUSH": "Agresivní útok na střed a rychlý střet",
            "HOLD": "Obranná formace a držení klíčových pozic",
            "SKIRMISH": "Průzkum, odstup a dálkové ostřelování",
        }.get(doc_name, "")
        f_name = FACTION_META[f_id]["name"]
        self.show_banner(f"[DOKTRINA] {f_name}: Doktrína přepnuta na [{doc_name}] - {doc_desc}", 3.5)

    def _trigger_button(self, key: str):
        self._btn_flash[key] = time.perf_counter()
        exact_handlers = {
            "back_menu": lambda: pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)),
            "play_pause": lambda: (
                setattr(self, "running", not self.running),
                self.show_banner("[PLAY] SIMULACE BĚŽÍ" if self.running else "[PAUZA] SIMULACE POZASTAVENA", 2.0),
            ),
            "next_epoch": self._on_next_epoch,
            "toggle_follow": lambda: (
                setattr(self, "follow_selected", not self.follow_selected),
                self.show_banner("Sledování kamery: " + ("ZAPNUTO [Kamera sleduje cíl]" if self.follow_selected else "VYPNUTO [Volná kamera]"), 2.0),
            ),
            "cycle_speed": lambda: (
                setattr(self, "speed_idx", (self.speed_idx + 1) % len(self.speed_presets)),
                self.show_banner(f"[BLESK] Rychlost: {self.speed_presets[self.speed_idx]:g}x", 2.0),
            ),
            "cycle_scenario": self._on_cycle_scenario,
            "cycle_units": lambda: (
                setattr(self, "unit_preset_idx", (self.unit_preset_idx + 1) % len(self.unit_presets)),
                self.reboot_simulation(),
                self.show_banner(f"Rozpočet kampaně přenastaven ({self.engine.budget_per_faction}b / frakce)", 3.0),
            ),
            "save_snap": self._on_save_snap,
            "load_snap": self._on_load_snap,
        }
        handler = exact_handlers.get(key)
        if handler is not None:
            handler()
            return
        if key.startswith("speed_"):
            self.speed_idx = int(key.split("_")[1])
            self.show_banner(f"[BLESK] Rychlost simulace nastavena: {self.speed_presets[self.speed_idx]:g}x", 2.0)
        elif key.startswith("doc_"):
            self._on_cycle_doctrine(int(key.split("_")[1]))

    def draw_button(self, surf, key: str, rect: pygame.Rect, label: str, active: bool = False, accent=ACCENT_CYAN):
        _draw_hud_button(
            surf, self.ui_buttons, self._btn_flash, self.text_cache, self.f_small,
            key=key, rect=rect, label=label, active=active, accent=accent,
        )

    def render_battlefield(self, state: dict, unit_info: dict | None):
        vr = self.viewport_rect
        pygame.draw.rect(self.screen, (10, 13, 19), vr)

        def to_s(x, y):
            return self.world_to_screen(x, y)

        r_scale = (vr.width / float(self.SIM_W)) * self.zoom

        # 1. Statické prvky mapy (sdílený renderer = stejný vzhled jako radar v launcheru)
        step_px = max(28, int(64 * self.zoom))
        if self.zoom <= 1.01:
            bf_render.draw_static_map(
                self.screen, vr, to_s, r_scale, state, self.faction_colors,
                mode="full", step_px=step_px,
            )
        else:
            bf_render.draw_grid(self.screen, vr, step_px=step_px, color=(15, 21, 30))
            bf_render.draw_bases(self.screen, state["bases"], to_s, r_scale, self.faction_colors)
            bf_render.draw_walls(self.screen, state["walls"], to_s, mode="full")
            bf_render.draw_pillars(self.screen, state["pillars"], to_s, r_scale, mode="full")
        bf_render.draw_gates(self.screen, state.get("gates", []), to_s, self.faction_colors, mode="full")
        bf_render.draw_nodes(self.screen, state["nodes"], to_s, r_scale, self.faction_colors)

        # 2. Efekty
        bf_render.draw_heals(self.screen, state, to_s)
        bf_render.draw_tracers(self.screen, state, to_s, self.faction_colors)
        bf_render.draw_sparks(self.screen, state, to_s, mode="full")
        bf_render.draw_lasers(self.screen, state, to_s)
        bf_render.draw_smoke(self.screen, state.get("smokes", []), to_s, r_scale)
        bf_render.draw_particles(self.screen, state, to_s, self.zoom)

        # 7. Zobrazení zorného pole a zaměření pro sledovanou jednotku
        if unit_info is not None and unit_info["hp"] > 0:
            ux, uy = self.world_to_screen(unit_info["pos"][0], unit_info["pos"][1])
            u_th = unit_info.get("heading", 0.0)
            u_range = unit_info.get("range", 200.0)
            sr_range = int((u_range / self.SIM_W) * vr.width * self.zoom)

            fov_half = math.radians(70)
            lx = ux + int(math.cos(u_th - fov_half) * sr_range)
            ly = uy + int(math.sin(u_th - fov_half) * sr_range)
            rx = ux + int(math.cos(u_th + fov_half) * sr_range)
            ry = uy + int(math.sin(u_th + fov_half) * sr_range)

            pygame.draw.line(self.screen, (60, 130, 200), (ux, uy), (lx, ly), 1)
            pygame.draw.line(self.screen, (60, 130, 200), (ux, uy), (rx, ry), 1)

            tar_pos = unit_info.get("target_pos")
            if tar_pos:
                tar_dx = tar_pos[0] - unit_info["pos"][0]
                tar_dy = tar_pos[1] - unit_info["pos"][1]
                tar_ang = math.atan2(tar_dy, tar_dx)
                tar_diff = (tar_ang - u_th + math.pi) % (2.0 * math.pi) - math.pi
                # Cíl se vykreslí pouze pokud je uvnitř zorného kuželu nebo v těsném dosahu
                if abs(tar_diff) < fov_half + 0.04 or math.hypot(tar_dx, tar_dy) < 75.0:
                    tx, ty = self.world_to_screen(tar_pos[0], tar_pos[1])
                    pygame.draw.line(self.screen, (255, 55, 55), (ux, uy), (tx, ty), 1)
                    pygame.draw.circle(self.screen, (255, 55, 55), (tx, ty), 7, width=1)
                    pygame.draw.circle(self.screen, (255, 220, 220), (tx, ty), 2)

        # 8. Vojáci – sdílené výrazné siluety (chevron / delta+kříž / diamant / šestiúhelník)
        tracked_id = unit_info["unit_idx"] if unit_info is not None else -1
        bf_render.draw_units(
            self.screen, state, to_s, r_scale, self.zoom,
            self.faction_colors, mode="full", tracked_id=tracked_id,
        )

        if self.status_banner_timer > 0.0 and self.status_banner:
            b_surf = self.f_sub.render(f" {self.status_banner} ", True, ACCENT_CYAN)
            b_rect = pygame.Rect(vr.x + 14, vr.bottom - 34, b_surf.get_width() + 16, 24)
            pygame.draw.rect(self.screen, (14, 22, 34), b_rect, border_radius=4)
            pygame.draw.rect(self.screen, ACCENT_CYAN, b_rect, width=1, border_radius=4)
            self.screen.blit(b_surf, (b_rect.x + 8, b_rect.y + 4))

        pygame.draw.rect(self.screen, BORDER_COL, vr, width=1)

    def render_all(self, state: dict, unit_info: dict | None):
        W, H = self.screen.get_size()
        self.screen.fill(BG_DARK)
        self.ui_buttons.clear()

        top_h = 56
        dock_w = max(480, min(560, int(W * 0.34)))
        vp_w = W - dock_w - 18
        vp_h = H - top_h - 14

        self.viewport_rect = pygame.Rect(10, top_h + 6, vp_w, vp_h)
        dock_rect = pygame.Rect(vp_w + 14, top_h + 6, dock_w - 4, vp_h)

        t_ui0 = time.perf_counter()
        self._draw_top_bar(W, top_h)
        t_ui1 = time.perf_counter()

        t_world0 = time.perf_counter()
        self.render_battlefield(state, unit_info)
        t_world1 = time.perf_counter()

        t_dock0 = time.perf_counter()
        self._draw_right_dock(dock_rect, unit_info, state)
        t_dock1 = time.perf_counter()

        render_prof.record("render_ui_ms", ((t_ui1 - t_ui0) + (t_dock1 - t_dock0)) * 1000.0)
        render_prof.record("render_world_ms", (t_world1 - t_world0) * 1000.0)

        t_flip0 = time.perf_counter()
        pygame.display.flip()
        t_flip1 = time.perf_counter()
        render_prof.record("flip_ms", (t_flip1 - t_flip0) * 1000.0)

    def _draw_top_bar(self, W: int, top_h: int):
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, W, top_h))
        pygame.draw.line(self.screen, BORDER_COL, (0, top_h - 1), (W, top_h - 1), 1)

        sec = int(self.engine.operation_time_sec)
        rem_sec = max(0, int(self.engine.MAX_OPERATION_DURATION - self.engine.operation_time_sec))
        time_str = f"{sec // 60:02d}:{sec % 60:02d} (Do evoluce: {rem_sec:02d}s)"

        # 1. Levá zóna: Kompaktní informační panel
        left_w = min(220, max(170, int(W * 0.17)))
        t1 = f"[BLESK] CYBER-TACTICS // EP #{self.engine.current_epoch:02d}"
        self.screen.blit(self.text_cache.render(self.f_sub, t1, ACCENT_CYAN), (10, 6))

        scen_name = SCENARIO_NAMES[self.engine.current_scenario_idx].split("(")[0].strip()
        t2 = f"{scen_name[:16]} • {time_str}"
        self.screen.blit(self.text_cache.render(self.f_tiny, t2, ACCENT_AMBER), (10, 23))

        fps_val = getattr(self, "measured_fps", 60.0)
        render_ms = getattr(self, "render_time_ms", 5.0)
        sim_ms = getattr(self, "sim_time_ms", 1.0)
        t_perf = f"FPS:{fps_val:3.0f} ({render_ms:3.1f}ms) | SIM:{sim_ms:3.1f}ms"
        self.screen.blit(self.text_cache.render(self.f_tiny, t_perf, ACCENT_GREEN), (10, 38))

        # 2. Pravá zóna: Rychlosti, Kamera a Play/Pause
        r_w = 330
        sp_x = W - r_w
        for s_i, sp_val in enumerate(self.speed_presets[:4]):
            bx = sp_x + s_i * 36
            self.draw_button(self.screen, f"speed_{s_i}", pygame.Rect(bx, 12, 32, 32), f"{sp_val:g}x", active=(self.speed_idx == s_i), accent=ACCENT_AMBER)

        self.draw_button(
            self.screen,
            "toggle_follow",
            pygame.Rect(sp_x + 4 * 36 + 2, 12, 64, 32),
            "KAMERA" if self.follow_selected else "VOLNÁ",
            active=self.follow_selected,
            accent=ACCENT_CYAN,
        )

        self.draw_button(
            self.screen,
            "play_pause",
            pygame.Rect(sp_x + 4 * 36 + 70, 12, 110, 32),
            "[PAUZA] PAUZA" if self.running else "[PLAY] TRÉNOVAT",
            active=self.running,
            accent=ACCENT_GREEN if self.running else ACCENT_AMBER,
        )

        # 3. Střední zóna: Navigační a akční tlačítka s čistými popisky
        btn_defs = []
        if self.embedded:
            btn_defs.append(("back_menu", "MENU [ESC]", True, ACCENT_PINK, 90))
        btn_defs.append(("cycle_scenario", "MAPA [1-3]", True, ACCENT_CYAN, 90))
        btn_defs.append(("next_epoch", "EVOLUCE [E]", True, ACCENT_GREEN, 95))
        btn_defs.append(("save_snap", "ULOŽIT [F5]", True, ACCENT_AMBER, 95))
        btn_defs.append(("load_snap", "NAČÍST [L]", False, ACCENT_CYAN, 85))

        btn_gap = 8
        tot_btns_w = sum(b[4] for b in btn_defs) + (len(btn_defs) - 1) * btn_gap
        avail_w = sp_x - 15 - (left_w + 15)
        start_bx = left_w + 15 + max(0, (avail_w - tot_btns_w) // 2)

        cur_bx = start_bx
        for (b_key, b_label, b_active, b_acc, b_width) in btn_defs:
            self.draw_button(self.screen, b_key, pygame.Rect(cur_bx, 12, b_width, 32), b_label, active=b_active, accent=b_acc)
            cur_bx += b_width + btn_gap

    def _draw_fitness_graph(self, rect: pygame.Rect, state: dict):
        """Vykreslí křivku učení (vývoj průměrné Fitness napříč epochami pro 4 frakce)."""
        pygame.draw.rect(self.screen, (13, 17, 25), rect, border_radius=3)
        pygame.draw.rect(self.screen, BORDER_COL, rect, width=1, border_radius=3)

        self.screen.blit(self.text_cache.render(self.f_tiny, "KŘIVKA UČENÍ AI (PRŮMĚRNÁ FITNESS)", ACCENT_CYAN), (rect.x + 6, rect.y + 4))

        hist = state.get("history_avg_fitness", [[] for _ in range(4)])
        max_len = max((len(h) for h in hist), default=0)
        if max_len < 1:
            msg = self.text_cache.render(self.f_tiny, "Čekám na dokončení 1. epochy [E]...", TEXT_DIM)
            self.screen.blit(msg, (rect.x + 8, rect.y + rect.height // 2))
            return

        all_vals = [v for h in hist for v in h[-25:]]
        min_v = min(all_vals) if all_vals else 0.0
        max_v = max(all_vals) if all_vals else 10.0
        span = max(5.0, max_v - min_v)

        gx = rect.x + 6
        gy = rect.y + 18
        gw = rect.width - 32
        gh = rect.height - 24

        lbl_max = self.text_cache.render(self.f_tiny, f"{max_v:.0f}", TEXT_DIM)
        lbl_min = self.text_cache.render(self.f_tiny, f"{min_v:.0f}", TEXT_DIM)
        self.screen.blit(lbl_max, (rect.right - lbl_max.get_width() - 4, gy))
        self.screen.blit(lbl_min, (rect.right - lbl_min.get_width() - 4, gy + gh - lbl_min.get_height()))

        for f in range(min(4, len(hist))):
            series = hist[f][-25:]
            if not series:
                continue
            col = FACTION_META[f]["color"]
            if len(series) == 1:
                py = int(gy + gh - ((series[0] - min_v) / span) * gh)
                pygame.draw.circle(self.screen, col, (gx + gw // 2, py), 3)
            else:
                pts = []
                for idx, val in enumerate(series):
                    px = int(gx + (idx / (len(series) - 1)) * gw)
                    py = int(gy + gh - ((val - min_v) / span) * gh)
                    pts.append((px, max(gy, min(gy + gh, py))))
                pygame.draw.lines(self.screen, col, False, pts, 2)

    def _draw_right_dock(self, rect: pygame.Rect, unit_info: dict | None, state: dict):
        _draw_right_dock(self, rect, unit_info, state)

    def run(self):
        while True:
            t_ev0 = time.perf_counter()
            ev_ok = self.handle_events()
            t_ev1 = time.perf_counter()
            render_prof.record("events_ms", (t_ev1 - t_ev0) * 1000.0)
            if not ev_ok:
                break

            if self.status_banner_timer > 0.0:
                self.status_banner_timer = max(0.0, self.status_banner_timer - (1.0 / 60.0))

            for msg in self.bridge.pop_banners():
                self.show_banner(msg, 3.0)

            # 180 FPS Lock-Free State Bridge & Async Simulation Worker
            if self.worker.is_running() != self.running:
                self.worker.set_paused(not self.running)
            self.worker.set_speed(self.speed_presets[self.speed_idx])

            t_bridge0 = time.perf_counter()
            state = self.bridge.get_interpolated_state()
            if state is None:
                state = getattr(self, "last_state", None)
            t_bridge1 = time.perf_counter()
            render_prof.record("bridge_lerp_ms", (t_bridge1 - t_bridge0) * 1000.0)
            if state is None:
                self.clock.tick(getattr(self, "target_fps", 60))
                continue
            self.last_state = state
            
            t_insp0 = time.perf_counter()
            unit_info = state.get("inspected_unit")
            t_insp1 = time.perf_counter()
            render_prof.record("inspect_unit_ms", (t_insp1 - t_insp0) * 1000.0)

            if unit_info is not None and self.follow_selected and unit_info["hp"] > 0:
                pos_arr = state.get("pos")
                uid = unit_info.get("unit_idx", -1)
                if pos_arr is not None and 0 <= uid < len(pos_arr):
                    ux, uy = pos_arr[uid]
                else:
                    ux, uy = unit_info["pos"]
                self.cam_x = 0.85 * self.cam_x + 0.15 * ux
                self.cam_y = 0.85 * self.cam_y + 0.15 * uy

            t_rend_start = time.perf_counter()
            self.render_all(state, unit_info)
            t_rend = time.perf_counter() - t_rend_start

            # Aktualizace systémové telemetrie výkonu
            self.sim_time_ms = getattr(self.worker, "step_time_ms", 1.0)
            self.render_time_ms = 0.85 * getattr(self, "render_time_ms", 1.0) + 0.15 * (t_rend * 1000.0)
            self.measured_fps = self.clock.get_fps()
            if self.device.type == "cuda":
                self.vram_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            else:
                self.vram_mb = 0.0

            # Kooperativní uvolnění procesoru operačnímu systému Windows (60 FPS zámek pro hladký GIL):
            t_tick0 = time.perf_counter()
            self.clock.tick(getattr(self, "target_fps", 60))
            t_tick1 = time.perf_counter()
            render_prof.record("clock_tick_ms", (t_tick1 - t_tick0) * 1000.0)

            if render_prof.should_report():
                render_prof.report_and_reset(f"FPS: {self.measured_fps:.1f}")

        if getattr(self, "_owns_worker", True):
            self.worker.stop()
        if hasattr(self.telemetry, "shutdown"):
            self.telemetry.shutdown(wait=True)
        self.telemetry.flush_spatial_parquet()
        if not self.embedded:
            pygame.quit()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default="", help="Cesta ke snapshotu .pt pro okamžité načtení")
    parser.add_argument("--scenario", type=int, default=-1, help="Počáteční scénář (0=Domination, 1=Citadel, 2=Chasm)")
    parser.add_argument("--units", type=int, default=0, help="Počáteční rozpočet jednotek (200, 400, 600, 800)")
    args, _ = parser.parse_known_args()

    app = OptimizedTacticalHUD()
    if args.units in app.unit_presets:
        app.unit_preset_idx = app.unit_presets.index(args.units)
        app.reboot_simulation()
    if args.scenario in (0, 1, 2):
        app.engine.switch_scenario(args.scenario)
    if args.load and os.path.exists(args.load):
        if app.telemetry.load_snapshot(app.engine, args.load):
            app.show_banner(f"Načten snapshot: {os.path.basename(args.load)}", 3.5)
    app.run()


if __name__ == "__main__":
    main()
