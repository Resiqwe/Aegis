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
from src.engine_core import RealisticBattleEngine, FACTION_COLORS, DOCTRINE_LIST
from src.ai_brains import CLASS_GRUNT, CLASS_MEDIC, CLASS_SNIPER, CLASS_JUGGERNAUT, TECH_NAMES, TECH_COSTS
from src.telemetry_logger import TelemetryManager
from src.engine.state_bridge import LockFreeStateBridge
from src.engine.async_runner import AsyncSimulationWorker
import battlefield_renderer as bf_render

FACTION_META = [
    {"name": "ALPHA VANGUARD", "short": "ALPHA", "color": (0, 240, 210)},
    {"name": "OMEGA SYNDICATE", "short": "OMEGA", "color": (255, 45, 115)},
    {"name": "SOLAR CORE", "short": "SOLAR", "color": (255, 195, 25)},
    {"name": "PHANTOM UNIT", "short": "PHANTOM", "color": (175, 80, 250)},
]

BG_DARK = (11, 14, 21)
PANEL_BG = (16, 21, 31)
PANEL_HEADER = (24, 32, 46)
BORDER_COL = (42, 56, 80)
ACCENT_CYAN = (0, 240, 210)
ACCENT_AMBER = (255, 195, 25)
ACCENT_GREEN = (65, 235, 100)
ACCENT_PINK = (255, 45, 120)
TEXT_MAIN = (235, 242, 250)
TEXT_DIM = (145, 160, 185)


def safe_rgb(col, default=(200, 200, 200)) -> tuple[int, int, int]:
    try:
        return (
            max(0, min(255, int(col[0]))),
            max(0, min(255, int(col[1]))),
            max(0, min(255, int(col[2]))),
        )
    except Exception:
        return default


class OptimizedTacticalHUD:
    def __init__(
        self,
        existing_screen: pygame.Surface | None = None,
        existing_engine: RealisticBattleEngine | None = None,
        existing_telemetry: TelemetryManager | None = None,
        embedded: bool = False,
    ):
        self.embedded = embedded
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
        self.target_fps = 180
        self._sim_accumulator = 0.0
        self._last_sim_time = time.perf_counter()
        self.target_fps = 180
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

        self.bridge = LockFreeStateBridge()
        self.worker = AsyncSimulationWorker(self.engine, self.bridge)
        self.worker.start()

        self.zoom = 1.0
        self.cam_x = self.SIM_W * 0.5
        self.cam_y = self.SIM_H * 0.5
        self.panning = False
        self.pan_start_mouse = (0, 0)
        self.pan_start_cam = (self.cam_x, self.cam_y)

        self.selected_unit = -1  # -1 = auto-track nejlepšího vojáka generace
        self.follow_selected = False
        self.ui_buttons = {}
        self._btn_flash = {}  # {key: timestamp} pro okamžitou vizuální odezvu stisku
        self.viewport_rect = pygame.Rect(0, 0, 100, 100)
        self.status_banner = ""
        self.status_banner_timer = 0.0

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
        self.selected_unit = -1

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
                mx, my = ev.pos
                if ev.button == 1:
                    if not self._check_ui_click(mx, my) and self.viewport_rect.collidepoint(mx, my):
                        wx, wy = self.screen_to_world(mx, my)
                        if hasattr(self, "last_state") and self.last_state and "pos" in self.last_state:
                            pos_np = self.last_state["pos"]
                        else:
                            pos_np = self.engine.pos.cpu().numpy()
                        dists = np.hypot(pos_np[:, 0] - wx, pos_np[:, 1] - wy)
                        min_d = np.min(dists)
                        if min_d <= 40.0:
                            self.selected_unit = int(np.argmin(dists))
                        else:
                            self.selected_unit = -1

                elif ev.button in (2, 3) and self.viewport_rect.collidepoint(mx, my):
                    self.panning = True
                    self.follow_selected = False
                    self.pan_start_mouse = (mx, my)
                    self.pan_start_cam = (self.cam_x, self.cam_y)

            elif ev.type == pygame.MOUSEBUTTONUP and ev.button in (2, 3):
                self.panning = False

            elif ev.type == pygame.MOUSEMOTION and self.panning and self.zoom > 1.01:
                mx, my = ev.pos
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

    def _trigger_button(self, key: str):
        self._btn_flash[key] = time.perf_counter()
        if key == "back_menu":
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        elif key == "play_pause":
            self.running = not self.running
            self.show_banner("[PLAY] SIMULACE BĚŽÍ" if self.running else "[PAUZA] SIMULACE POZASTAVENA", 2.0)
        elif key == "next_epoch":
            if hasattr(self, "worker") and self.worker is not None:
                self.worker.post_command("force_evolve")
            else:
                self.engine.finish_operation_and_evolve()
            self.show_banner(f"Epocha #{self.engine.current_epoch:02d}: Požadavek na novou epochu odeslán", 3.0)
        elif key.startswith("speed_"):
            self.speed_idx = int(key.split("_")[1])
            self.show_banner(f"[BLESK] Rychlost simulace nastavena: {self.speed_presets[self.speed_idx]:g}x", 2.0)
        elif key == "toggle_follow":
            self.follow_selected = not self.follow_selected
            self.show_banner("Sledování kamery: " + ("ZAPNUTO [Kamera sleduje cíl]" if self.follow_selected else "VYPNUTO [Volná kamera]"), 2.0)
        elif key == "cycle_speed":
            self.speed_idx = (self.speed_idx + 1) % len(self.speed_presets)
            self.show_banner(f"[BLESK] Rychlost: {self.speed_presets[self.speed_idx]:g}x", 2.0)
        elif key == "cycle_scenario":
            next_scen = (self.engine.current_scenario_idx + 1) % 3
            if hasattr(self, "worker") and self.worker is not None:
                self.worker.post_command("switch_scenario", next_scen)
            else:
                self.engine.switch_scenario(next_scen)
            scen_name = SCENARIO_NAMES[next_scen].split('(')[0].strip()
            self.show_banner(f"[MAPA] Scénář přepnut: {scen_name}", 3.0)
        elif key == "cycle_units":
            self.unit_preset_idx = (self.unit_preset_idx + 1) % len(self.unit_presets)
            self.reboot_simulation()
            self.show_banner(f"Rozpočet kampaně přenastaven ({self.engine.budget_per_faction}b / frakce)", 3.0)
        elif key == "save_snap":
            path = self.telemetry.save_snapshot(self.engine)
            self.show_banner(f"[ULOZIT] Uložen model: {os.path.basename(path)}", 3.0)
        elif key == "load_snap":
            snaps = self.telemetry.list_available_snapshots()
            if snaps and self.telemetry.load_snapshot(self.engine, snaps[0]):
                self.show_banner(f"[NACIST] Načten model: {os.path.basename(snaps[0])}", 3.0)
            else:
                self.show_banner("[VAROVANI] Žádný snapshot k načtení nenalezen!", 3.0)
        elif key.startswith("doc_"):
            f_id = int(key.split("_")[1])
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
                "SKIRMISH": "Průzkum, odstup a dálkové ostřelování"
            }.get(doc_name, "")
            f_name = FACTION_META[f_id]['name']
            self.show_banner(f"[DOKTRINA] {f_name}: Doktrína přepnuta na [{doc_name}] - {doc_desc}", 3.5)

    def draw_button(self, surf, key: str, rect: pygame.Rect, label: str, active: bool = False, accent=ACCENT_CYAN):
        self.ui_buttons[key] = rect
        mx, my = pygame.mouse.get_pos()
        hover = rect.collidepoint(mx, my)

        acc = safe_rgb(accent)
        # Efekt stisku tlačítka (flash 140 ms po kliknutí)
        is_flash = (time.perf_counter() - self._btn_flash.get(key, 0.0)) < 0.14

        if is_flash:
            bg = (min(255, int(acc[0] * 0.55 + 50)), min(255, int(acc[1] * 0.55 + 50)), min(255, int(acc[2] * 0.55 + 50)))
            border = (255, 255, 255)
            txt_col = (255, 255, 255)
        elif active:
            bg = (int(acc[0] * 0.25), int(acc[1] * 0.25), int(acc[2] * 0.25))
            border = acc
            txt_col = acc
        elif hover:
            bg = (34, 46, 66)
            border = (140, 180, 230)
            txt_col = TEXT_MAIN
        else:
            bg = (18, 24, 34)
            border = BORDER_COL
            txt_col = TEXT_DIM

        pygame.draw.rect(surf, bg, rect, border_radius=4)
        pygame.draw.rect(surf, border, rect, width=1 if not (active or is_flash) else 2, border_radius=4)
        txt = self.f_small.render(label, True, txt_col)
        surf.blit(txt, (rect.x + (rect.width - txt.get_width()) // 2, rect.y + (rect.height - txt.get_height()) // 2))

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

        self._draw_top_bar(W, top_h)
        self.render_battlefield(state, unit_info)
        self._draw_right_dock(dock_rect, unit_info, state)

        pygame.display.flip()

    def _draw_top_bar(self, W: int, top_h: int):
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, W, top_h))
        pygame.draw.line(self.screen, BORDER_COL, (0, top_h - 1), (W, top_h - 1), 1)

        sec = int(self.engine.operation_time_sec)
        rem_sec = max(0, int(self.engine.MAX_OPERATION_DURATION - self.engine.operation_time_sec))
        time_str = f"{sec // 60:02d}:{sec % 60:02d} (Do evoluce: {rem_sec:02d}s)"

        # 1. Levá zóna: Kompaktní informační panel
        left_w = min(220, max(170, int(W * 0.17)))
        t1 = f"[BLESK] CYBER-TACTICS // EP #{self.engine.current_epoch:02d}"
        self.screen.blit(self.f_sub.render(t1, True, ACCENT_CYAN), (10, 6))

        scen_name = SCENARIO_NAMES[self.engine.current_scenario_idx].split("(")[0].strip()
        t2 = f"{scen_name[:16]} • {time_str}"
        self.screen.blit(self.f_tiny.render(t2, True, ACCENT_AMBER), (10, 23))

        fps_val = getattr(self, "measured_fps", 60.0)
        render_ms = getattr(self, "render_time_ms", 5.0)
        sim_ms = getattr(self, "sim_time_ms", 1.0)
        t_perf = f"FPS:{fps_val:3.0f} ({render_ms:3.1f}ms) | SIM:{sim_ms:3.1f}ms"
        self.screen.blit(self.f_tiny.render(t_perf, True, ACCENT_GREEN), (10, 38))

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

        self.screen.blit(self.f_tiny.render("KŘIVKA UČENÍ AI (PRŮMĚRNÁ FITNESS)", True, ACCENT_CYAN), (rect.x + 6, rect.y + 4))

        hist = state.get("history_avg_fitness", [[] for _ in range(4)])
        max_len = max((len(h) for h in hist), default=0)
        if max_len < 1:
            msg = self.f_tiny.render("Čekám na dokončení 1. epochy [E]...", True, TEXT_DIM)
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

        lbl_max = self.f_tiny.render(f"{max_v:.0f}", True, TEXT_DIM)
        lbl_min = self.f_tiny.render(f"{min_v:.0f}", True, TEXT_DIM)
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
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=5)
        pygame.draw.rect(self.screen, BORDER_COL, rect, width=1, border_radius=5)

        pad = 8
        x = rect.x + pad
        w = rect.width - 2 * pad
        y = rect.y + pad

        # ==========================================
        # 1. VOJENSKÁ KAMPAŇ, VELITELÉ & ROI UČENÍ
        # ==========================================
        hdr_h = max(22, int(22 * self.ui_scale))
        pygame.draw.rect(self.screen, PANEL_HEADER, (x, y, w, hdr_h), border_radius=3)
        self.screen.blit(self.f_sub.render("1. FRAKCE, GENERÁLOVÉ (ROI) & DOKTRÍNY", True, ACCENT_CYAN), (x + 8, y + max(2, (hdr_h - self.f_sub.get_height()) // 2)))
        y += hdr_h + 3

        alive_mask = state["alive"]
        teams = state["teams"]
        kills = self.engine.kills_per_faction.cpu().numpy()
        wins = self.engine.epoch_wins_per_faction
        alive_counts = [int(np.sum((teams == f) & alive_mask)) for f in range(self.engine.n_factions)]
        total_alive = max(1, sum(alive_counts))

        # Tug-of-war barevný pruh poměru sil
        bar_h = 6
        cur_bx = x
        for f in range(self.engine.n_factions):
            seg_w = int((alive_counts[f] / total_alive) * w)
            if seg_w > 0:
                pygame.draw.rect(self.screen, FACTION_META[f]["color"], (cur_bx, y, seg_w, bar_h))
                cur_bx += seg_w
        pygame.draw.rect(self.screen, BORDER_COL, (x, y, w, bar_h), width=1)
        y += 9

        generals_info = state.get("generals", [])
        for f in range(self.engine.n_factions):
            f_col = FACTION_META[f]["color"]
            f_alive = alive_counts[f]
            init_c = max(1, self.engine.initial_team_counts[f] if f < len(self.engine.initial_team_counts) else 100)
            eff_pct = int((f_alive / init_c) * 100)

            gen = generals_info[f] if f < len(generals_info) else {}
            lvl = gen.get("level", 0)
            xp = gen.get("xp", 0.0)
            comp = gen.get("composition", [100, 0, 0, 0])
            probs = gen.get("probs", [1.0, 0.0, 0.0, 0.0])

            row_h = max(56, int(58 * self.ui_scale))
            row = pygame.Rect(x, y, w, row_h)
            pygame.draw.rect(self.screen, (20, 26, 38), row, border_radius=4)
            pygame.draw.rect(self.screen, BORDER_COL, row, width=1, border_radius=4)
            pygame.draw.rect(self.screen, f_col, (x, y, 4, row_h), border_radius=2)

            # Doctrine button na pravé straně s vertikálním vycentrováním
            doc_btn_w = max(74, int(80 * self.ui_scale))
            doc_btn_h = max(26, int(28 * self.ui_scale))
            doc_btn_y = y + (row_h - doc_btn_h) // 2
            cur_doc = self.engine.faction_doctrines[f]
            d_name = DOCTRINE_LIST[cur_doc]
            doc_colors = [
                (140, 155, 175),   # BALANCED (neutrální ocelová)
                (255, 75, 75),     # RUSH (agresivní červená)
                (60, 160, 255),    # HOLD (obranná modrá)
                (200, 100, 255),   # SKIRMISH (taktická fialová)
            ]
            doc_accent = doc_colors[cur_doc]
            self.draw_button(
                self.screen,
                f"doc_{f}",
                pygame.Rect(x + w - doc_btn_w - 6, doc_btn_y, doc_btn_w, doc_btn_h),
                d_name,
                active=True,
                accent=doc_accent,
            )

            # Texty s čistým vertikálním a horizontálním paddingem
            t_line1 = f"{FACTION_META[f]['short']} L{lvl} W:{wins[f]} | Síla: {eff_pct}% ({f_alive}) | K: {kills[f]}"
            self.screen.blit(self.f_small.render(t_line1, True, f_col if eff_pct > 22 else (230, 50, 50)), (x + 12, y + 6))

            # Aktivní Power-Upy frakce
            f_techs = state.get("faction_techs", [[] for _ in range(4)])
            tech_tags = []
            if f < len(f_techs):
                tag_map = {0: "NANIT", 1: "RADAR", 2: "ŠTÍT", 3: "OVERCLOCK", 4: "AEROSOL"}
                for tid in f_techs[f]:
                    tech_tags.append(tag_map.get(tid, f"T{tid}"))
            tech_str = (" [" + "][".join(tech_tags) + "]") if tech_tags else " [BEZ TECH]"

            comp_txt = (
                f"P:{comp[0]} M:{comp[1]} S:{comp[2]} J:{comp[3]}  •  "
                f"Váhy: {int(probs[0]*100)}/{int(probs[1]*100)}/{int(probs[2]*100)}/{int(probs[3]*100)}%"
            )
            self.screen.blit(self.f_tiny.render(comp_txt, True, TEXT_DIM), (x + 12, y + 24))
            t_tech_surf = self.f_tiny.render(f"TECH:{tech_str}", True, ACCENT_AMBER if tech_tags else TEXT_DIM)
            self.screen.blit(t_tech_surf, (x + 12, y + 38))

            y += row_h + 4

        y += 3
        # ==========================================
        # 2. KŘIVKA UČENÍ AI + BOJOVÝ PROTOKOL UDÁLOSTÍ
        # ==========================================
        pygame.draw.rect(self.screen, PANEL_HEADER, (x, y, w, hdr_h), border_radius=3)
        self.screen.blit(self.f_sub.render("2. VÝVOJ UČENÍ (FITNESS) & PROTOKOL", True, ACCENT_CYAN), (x + 8, y + max(2, (hdr_h - self.f_sub.get_height()) // 2)))
        y += hdr_h + 3

        mid_h = max(92, int(96 * self.ui_scale))
        graph_w = int(w * 0.46)
        feed_w = w - graph_w - 6

        self._draw_fitness_graph(pygame.Rect(x, y, graph_w, mid_h), state)

        feed_rect = pygame.Rect(x + graph_w + 6, y, feed_w, mid_h)
        pygame.draw.rect(self.screen, (13, 17, 25), feed_rect, border_radius=3)
        pygame.draw.rect(self.screen, BORDER_COL, feed_rect, width=1, border_radius=3)

        events = state.get("events", [])
        ey = y + 4
        ev_step = max(16, int(17 * self.ui_scale))
        for ev in events[-5:]:
            e_col = FACTION_META[ev["team"]]["color"] if 0 <= ev["team"] < 4 else ACCENT_AMBER
            ev_txt = ev["text"]
            if len(ev_txt) > 28:
                ev_txt = ev_txt[:26] + ".."
            self.screen.blit(self.f_tiny.render(ev_txt, True, e_col), (feed_rect.x + 6, ey))
            ey += ev_step
        y += mid_h + 8

        # ==========================================
        # 3. KOMPLETNÍ RENTGEN NEURONOVÉ SÍTĚ VOJÁKA (14 -> 28 -> 4)
        # ==========================================
        pygame.draw.rect(self.screen, PANEL_HEADER, (x, y, w, hdr_h), border_radius=3)
        lbl_mode = "MANUÁL" if self.selected_unit >= 0 else "AUTO-ELITA"
        self.screen.blit(self.f_sub.render(f"3. HIERARCHICKÝ MOZEK (14->64->64->7) // {lbl_mode}", True, ACCENT_CYAN), (x + 8, y + max(2, (hdr_h - self.f_sub.get_height()) // 2)))
        y += hdr_h + 3

        sens_step = max(13, int(14 * self.ui_scale))
        act_step = max(16, int(18 * self.ui_scale))
        insp_h = max(260, int(48 + 14 * sens_step + 12))
        insp_rect = pygame.Rect(x, y, w, insp_h)
        pygame.draw.rect(self.screen, (13, 17, 25), insp_rect, border_radius=4)
        pygame.draw.rect(self.screen, BORDER_COL, insp_rect, width=1, border_radius=4)

        if unit_info is None:
            msg = self.f_reg.render("Žádná jednotka není aktivní.", True, TEXT_DIM)
            self.screen.blit(msg, (x + 14, y + 30))
            return

        f_id = unit_info["team"]
        f_col = FACTION_META[f_id]["color"]
        u_title = (
            f"VOJÁK #{unit_info['unit_idx']:03d} (Slot #{unit_info.get('brain_slot', 0):03d}) "
            f"[{FACTION_META[f_id]['short']}] • {unit_info['class_name']}"
        )
        self.screen.blit(self.f_small.render(u_title, True, f_col), (x + 8, y + 4))

        st_parts = []
        if unit_info.get("unit_state_name"):
            st_parts.append(f"[{unit_info['unit_state_name']}]")
        if unit_info["in_cover"]:
            st_parts.append("[KRYT]")
        if unit_info["reloading"]:
            st_parts.append("[PŘEBÍJÍ]")
        if unit_info["suppression"] > 0.3:
            st_parts.append("[POTLAČEN]")
        status_lbl = " ".join(st_parts) if st_parts else "[AKTIVNÍ]"
        stat_color = ACCENT_AMBER if st_parts else TEXT_DIM

        fit_val = unit_info.get("fitness", 0.0)
        dmg_dealt = unit_info.get("damage_dealt", 0.0)
        dmg_taken = unit_info.get("damage_taken", 0.0)
        heal_done = unit_info.get("healing_done", 0.0)
        heal_str = f" Heal:{heal_done:.0f}" if heal_done > 0 else ""

        u_stats1 = (
            f"HP:{unit_info['hp']:.0f}/{unit_info['max_hp']:.0f} | "
            f"Zás:{unit_info['ammo']:.0f}/{unit_info['max_ammo']:.0f} | "
            f"Fit:{fit_val:.1f}  {status_lbl}"
        )
        self.screen.blit(self.f_tiny.render(u_stats1, True, stat_color), (x + 8, y + 19))

        u_stats2 = f"Kills:{unit_info.get('kills', 0)} | Dmg:{dmg_dealt:.0f} | Utrž:{dmg_taken:.0f}{heal_str}"
        self.screen.blit(self.f_tiny.render(u_stats2, True, TEXT_DIM), (x + 8, y + 33))

        df = unit_info["decision_flow"]
        col_w = (w - 14) // 2
        col_y_start = y + 48

        # Všech 14 senzorů v levém sloupci (Egocentrické tělesné vnímání)
        self.screen.blit(self.f_small.render("VSTUPNÍ SENZORY (14)", True, ACCENT_CYAN), (x + 6, col_y_start))
        
        SHORT_SENSORS = [
            "1.Cíl Vpřed", "2.Cíl Vpravo", "3.Blízkost",
            "4.Palba Vpř.", "5.Palba Vpr.",
            "6.Povel Vpř.", "7.Povel Vpr.",
            "8.Potlačení", "9.Zásobník", "10.Zdraví(HP)",
            "11.Kryt Vpř.", "12.Kryt Vpr.",
            "13.Koheze čety", "14.Stav zóny"
        ]

        bar_w = max(42, int(48 * self.ui_scale))
        for i, (name, val) in enumerate(df["sensors"][:14]):
            ry = col_y_start + 16 + i * sens_step
            s_name = SHORT_SENSORS[i] if i < len(SHORT_SENSORS) else name[:12]
            self.screen.blit(self.f_tiny.render(s_name, True, TEXT_DIM), (x + 6, ry))
            
            bx = x + col_w - bar_w - 4
            pygame.draw.rect(self.screen, (24, 32, 46), (bx, ry + 1, bar_w, 8), border_radius=2)
            half_b = bar_w // 2
            w_b = int(min(1.0, abs(val)) * (half_b - 1))
            c_b = ACCENT_CYAN if val >= 0 else ACCENT_PINK
            if val >= 0:
                pygame.draw.rect(self.screen, c_b, (bx + half_b, ry + 1, w_b, 8), border_radius=2)
            else:
                pygame.draw.rect(self.screen, c_b, (bx + half_b - w_b, ry + 1, w_b, 8), border_radius=2)

        # Pravý sloupec: 4 výstupní akce + 28 skrytých neuronů
        rx = x + col_w + 10
        self.screen.blit(self.f_small.render("VÝSTUPNÍ AKCE (7)", True, ACCENT_GREEN), (rx + 4, col_y_start))
        
        SHORT_ACTIONS = [
            "1.Surge(Plyn)", "2.Sway(Úkrok)", "3.Yaw(Míření)", "4.Trigger(Palba)",
            "5.Stance(Sprint)", "6.Ability(Doved.)", "7.Ping(Četa)"
        ]

        act_step_7 = max(12, int(13 * self.ui_scale))
        for i, (name, val) in enumerate(df["actions"][:7]):
            ry = col_y_start + 15 + i * act_step_7
            a_name = SHORT_ACTIONS[i] if i < len(SHORT_ACTIONS) else name[:12]
            self.screen.blit(self.f_tiny.render(a_name, True, TEXT_MAIN if abs(val) > 0.20 else TEXT_DIM), (rx + 4, ry))
            
            bx = rx + col_w - bar_w - 4
            pygame.draw.rect(self.screen, (24, 32, 46), (bx, ry + 1, bar_w, 7), border_radius=2)
            half_b = bar_w // 2
            w_b = int(min(1.0, abs(val)) * (half_b - 1))
            c_b = ACCENT_GREEN if val >= 0 else ACCENT_AMBER
            if val >= 0:
                pygame.draw.rect(self.screen, c_b, (bx + half_b, ry + 1, w_b, 7), border_radius=2)
            else:
                pygame.draw.rect(self.screen, c_b, (bx + half_b - w_b, ry + 1, w_b, 7), border_radius=2)

        # Dvouvrstvá hierarchická reprezentace: H1 (Percepce 64n) + H2 (Taktika 64n) = 128 neuronů
        hy = col_y_start + 15 + 7 * act_step_7 + 8
        self.screen.blit(self.f_tiny.render("H1: PERCEPCE PROSTORU (64 NEURONŮ // 4x16)", True, ACCENT_CYAN), (rx + 4, hy))
        h1_vals = df.get("hidden_h1", df.get("hidden", [])[:64])
        n_cols = 16
        cell_gap = 2
        cell_w = max(6, (col_w - (n_cols - 1) * cell_gap - 8) // n_cols)
        cell_h = max(6, int(6.5 * self.ui_scale))

        for idx, h_val in enumerate(h1_vals[:64]):
            r_idx = idx // n_cols
            c_idx = idx % n_cols
            cx = rx + 4 + c_idx * (cell_w + cell_gap)
            cy = hy + 14 + r_idx * (cell_h + cell_gap)
            val_clamped = 0.0 if (math.isnan(h_val) or math.isinf(h_val)) else max(0.0, min(1.0, abs(h_val)))
            inten = int(val_clamped * 200)
            col_h = (20, min(255, 45 + inten), 200) if h_val >= 0 else (min(255, 50 + inten), 25, 90)
            pygame.draw.rect(self.screen, col_h, (cx, cy, cell_w, cell_h), border_radius=1)
            pygame.draw.rect(self.screen, BORDER_COL, (cx, cy, cell_w, cell_h), width=1, border_radius=1)

        # Vrstva H2: Taktická syntéza
        h2_y = hy + 14 + 4 * (cell_h + cell_gap) + 6
        self.screen.blit(self.f_tiny.render("H2: TAKTICKÁ SYNTÉZA (64 NEURONŮ // 4x16)", True, ACCENT_AMBER), (rx + 4, h2_y))
        h2_vals = df.get("hidden_h2", df.get("hidden", [])[64:128])
        for idx, h_val in enumerate(h2_vals[:64]):
            r_idx = idx // n_cols
            c_idx = idx % n_cols
            cx = rx + 4 + c_idx * (cell_w + cell_gap)
            cy = h2_y + 14 + r_idx * (cell_h + cell_gap)
            val_clamped = 0.0 if (math.isnan(h_val) or math.isinf(h_val)) else max(0.0, min(1.0, abs(h_val)))
            inten = int(val_clamped * 200)
            col_h = (min(255, 30 + inten), min(255, 140 + inten // 2), 255) if h_val >= 0 else (min(255, 180 + inten // 3), 40, 70)
            pygame.draw.rect(self.screen, col_h, (cx, cy, cell_w, cell_h), border_radius=1)
            pygame.draw.rect(self.screen, BORDER_COL, (cx, cy, cell_w, cell_h), width=1, border_radius=1)

        y += insp_h + 8

        # ==========================================
        # 4. ŽIVÁ EKONOMIKA TŘÍD (ROI) & STATISTIKA BITVY (Využití prostoru)
        # ==========================================
        sec4_h = rect.bottom - y - 4
        if sec4_h >= 110:
            pygame.draw.rect(self.screen, PANEL_HEADER, (x, y, w, 22), border_radius=3)
            self.screen.blit(self.f_sub.render("4. EKONOMIKA TŘÍD (ROI) & STATISTIKY BOJE", True, ACCENT_CYAN), (x + 8, y + 4))
            y += 24

            sec4_rect = pygame.Rect(x, y, w, sec4_h - 24)
            pygame.draw.rect(self.screen, (13, 17, 25), sec4_rect, border_radius=4)
            pygame.draw.rect(self.screen, BORDER_COL, sec4_rect, width=1, border_radius=4)

            # Tabulka ROI tříd za tuto bitvu
            th_y = y + 5
            self.screen.blit(self.f_mono.render("TŘÍDA       CENA   ŽIVÝCH    UDĚLENO  UTRŽENO  KILLS   ROI/1b", True, ACCENT_AMBER), (x + 10, th_y))

            class_colors = [(74, 144, 226), (65, 235, 100), (255, 195, 25), (231, 76, 60)]
            stats = state.get("live_class_stats", [])
            for i, cs in enumerate(stats):
                ry = th_y + 18 + i * 17
                c_col = class_colors[i % len(class_colors)]
                c_name = cs["name"]
                c_cost = cs["cost"]
                c_al = cs["alive"]
                c_tot = cs["total"]
                c_dmg = cs["damage"]
                c_taken = cs["damage_taken"]
                c_k = cs["kills"]
                c_roi = cs["roi"]

                extra_txt = ""
                if cs["class_id"] == CLASS_MEDIC and cs["healing"] > 0:
                    extra_txt = f" +{cs['healing']:.0f}H"

                row_txt = f"{c_name:<11} {c_cost} b   {c_al:2d}/{c_tot:2d}    {c_dmg:6.0f}   {c_taken:6.0f}    {c_k:2d}    {c_roi:5.1f}{extra_txt}"
                self.screen.blit(self.f_mono.render(row_txt, True, c_col), (x + 10, ry))

            # Strategické body uzlů
            node_pts = state.get("node_points", [0.0, 0.0, 0.0, 0.0])
            bottom_y = sec4_rect.bottom - 17
            if bottom_y > th_y + 16 + 4 * 17 + 4:
                pts_str = f"Uzly (body): ALPHA:{node_pts[0]:.0f} | OMEGA:{node_pts[1]:.0f} | SOLAR:{node_pts[2]:.0f} | PHANTOM:{node_pts[3]:.0f}"
                self.screen.blit(self.f_tiny.render(pts_str, True, TEXT_DIM), (x + 8, bottom_y))

    def run(self):
        while True:
            if not self.handle_events():
                break

            if self.status_banner_timer > 0.0:
                self.status_banner_timer = max(0.0, self.status_banner_timer - (1.0 / 60.0))

            # 180 FPS Lock-Free State Bridge & Async Simulation Worker
            if self.worker.is_running() != self.running:
                self.worker.set_paused(not self.running)
            self.worker.set_speed(self.speed_presets[self.speed_idx])

            state = self.bridge.get_interpolated_state()
            if state is None:
                state = self.engine.get_battle_state_snapshot()
            self.last_state = state
            
            unit_info = None
            try:
                unit_info = self.engine.inspect_unit(self.selected_unit)
            except Exception:
                unit_info = None

            if unit_info is not None and self.follow_selected and unit_info["hp"] > 0:
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

            # Kooperativní uvolnění procesoru operačnímu systému Windows (eliminuje 100% zátěž jádra a sekání ostatních aplikací):
            self.clock.tick(getattr(self, "target_fps", 180))

        self.worker.stop()
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
