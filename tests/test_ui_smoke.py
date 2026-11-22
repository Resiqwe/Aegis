"""
tests/test_ui_smoke.py - Off-screen kouřový test UI komponent (Phase 3).

Ověřuje, že sdílené widgety, HUD inspektor i panely launcheru se korektně
importují a vykreslí 1 snímek do off-screen pygame.Surface bez pádu.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
for p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from engine_core import RealisticBattleEngine  # noqa: E402
from src.ui.hud_inspector import draw_right_dock  # noqa: E402
from src.ui.launcher_panels import draw_modern_analytics, draw_modern_dock, draw_modern_radar  # noqa: E402
from src.ui.widgets import ModernTextCache, draw_card, draw_hud_button, draw_launcher_button, safe_rgb  # noqa: E402


def _make_fonts():
    pygame.init()
    f_title = pygame.font.SysFont("consolas", 14, bold=True)
    f_sub = pygame.font.SysFont("consolas", 13, bold=True)
    f_small = pygame.font.SysFont("consolas", 12)
    f_tiny = pygame.font.SysFont("consolas", 11)
    f_reg = pygame.font.SysFont("consolas", 12)
    f_mono = pygame.font.SysFont("consolas", 11)
    return f_title, f_sub, f_small, f_tiny, f_reg, f_mono


def test_ui_offscreen_render_smoke():
    f_title, f_sub, f_small, f_tiny, f_reg, f_mono = _make_fonts()
    surf = pygame.Surface((1280, 720))
    cache = ModernTextCache()

    assert safe_rgb((300, -10, 128)) == (255, 0, 128)
    assert safe_rgb(None) == (200, 200, 200)

    engine = RealisticBattleEngine()
    engine.visual_mode = False
    engine.configure(n_units=48, n_factions=4)
    engine.step(1.0 / 60.0)
    state = engine.get_battle_state_snapshot()
    unit_info = engine.inspect_unit(0)

    # 1. HUD Right Dock smoke test
    hud_buttons = {}
    hud_flash = {}
    hud_stub = SimpleNamespace(
        screen=surf,
        ui_scale=1.0,
        text_cache=cache,
        f_sub=f_sub,
        f_small=f_small,
        f_tiny=f_tiny,
        f_reg=f_reg,
        f_mono=f_mono,
        engine=engine,
        selected_unit=-1,
        draw_button=lambda s, k, r, l, active=False, accent=(0, 240, 210): draw_hud_button(
            s, hud_buttons, hud_flash, cache, f_small, k, r, l, active=active, accent=accent
        ),
        _draw_fitness_graph=lambda rect, st: pygame.draw.rect(surf, (13, 17, 25), rect),
    )
    draw_right_dock(hud_stub, pygame.Rect(820, 40, 450, 670), unit_info, state)
    assert "doc_0" in hud_buttons

    # 2. Launcher Panels smoke test (dock + radar + heatmap + analytics)
    l_buttons = {}
    app_stub = SimpleNamespace(
        screen=surf,
        scale=1.0,
        text_cache=cache,
        f_title=f_title,
        f_sub=f_sub,
        f_small=f_small,
        f_tiny=f_tiny,
        engine=engine,
        sim_active=True,
        sim_paused=False,
        sim_status_line="OK",
        speed_presets=["1x", "2x", "4x", "8x", "16x", "MAX"],
        sim_speed_idx=0,
        start_from_snapshot=False,
        selected_scenario=0,
        selected_units_idx=0,
        snapshots=[],
        snapshot_rects=[],
        snapshot_box_rect=None,
        snapshot_scroll_offset=0,
        selected_snapshot_idx=0,
        view_mode="RADAR",
        heatmap_grid=np.zeros((22, 36), dtype=np.int32),
        epochs_data=[],
        draw_card=lambda r, bg=(15, 19, 28), border=(30, 40, 58), radius=8: draw_card(surf, r, bg=bg, border=border, radius=radius),
        draw_button=lambda k, r, t, subtitle="", active=False, accent=(0, 240, 210), radius=6, disabled=False: draw_launcher_button(
            surf, l_buttons, cache, f_sub, f_tiny, k, r, t, subtitle=subtitle, active=active, accent=accent, radius=radius, disabled=disabled
        ),
    )
    draw_modern_dock(app_stub, pygame.Rect(10, 50, 320, 650), state)
    draw_modern_radar(app_stub, pygame.Rect(340, 50, 540, 650), state)
    app_stub.view_mode = "HEATMAP"
    draw_modern_radar(app_stub, pygame.Rect(340, 50, 540, 650), state)
    draw_modern_analytics(app_stub, pygame.Rect(890, 50, 380, 650), state)
    assert "open_preview" in l_buttons
    assert "tab_radar" in l_buttons
