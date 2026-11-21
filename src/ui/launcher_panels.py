"""
ui/launcher_panels.py - Panely pro CyberTacticsLauncher (levý dok, Holo-Radar, heatmapa, analytika).
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy as np
import pygame

import battlefield_renderer as bf_render
import scenarios
from src.ui.widgets import (
    ACCENT_AMBER,
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_ROSE,
    BORDER_SUBTLE,
    CARD_BG,
    CLASS_COLORS,
    FACTION_THEMES,
    PANEL_SURFACE,
    TEXT_BODY,
    TEXT_HEAD,
    TEXT_MUTED,
)

if TYPE_CHECKING:
    from launcher import CyberTacticsLauncher


def draw_modern_dock(app: CyberTacticsLauncher, rect: pygame.Rect, state: dict | None = None) -> None:
    """Levý řídící dok: Ovládání běhu, parametry mapy a správa modelů."""
    if state is None:
        return
    app.draw_card(rect, bg=PANEL_SURFACE, border=BORDER_SUBTLE, radius=8)

    x = rect.x + 14
    w = rect.width - 28
    y = rect.y + 14

    # Sekce 1: Řízení operace
    t_sec1 = app.text_cache.render(app.f_title, "1. ŘÍZENÍ OPERACE", ACCENT_CYAN)
    app.screen.blit(t_sec1, (x, y))
    y += int(26 * app.scale)

    box_h = max(64, int(72 * app.scale))
    app.draw_card(pygame.Rect(x, y, w, box_h), bg=CARD_BG, border=BORDER_SUBTLE, radius=6)

    if app.sim_active:
        status_txt = "⏸ POZASTAVENO" if app.sim_paused else "🟢 KONTINUÁLNÍ SIMULACE BĚŽÍ"
        st_col = ACCENT_AMBER if app.sim_paused else ACCENT_GREEN
        app.screen.blit(app.text_cache.render(app.f_sub, status_txt, st_col), (x + 10, y + 8))

        alive_counts = state.get("alive_counts", [0, 0, 0, 0])
        ep_num = state.get("current_epoch", app.engine.current_epoch)
        t_sec = int(state.get("operation_time_sec", app.engine.operation_time_sec))
        alive_cnt = sum(alive_counts) if len(alive_counts) > 0 else 0
        tot_cnt = state.get("N", getattr(app.engine, "N", 400))
        scen_idx = state.get("scenario_idx", app.engine.current_scenario_idx)
        scen_name = scenarios.SCENARIO_NAMES[scen_idx].split("(")[0].strip()
        app.screen.blit(
            app.text_cache.render(
                app.f_small,
                f"Epocha #{ep_num:02d} • {t_sec//60:02d}:{t_sec%60:02d} • Živých: {alive_cnt}/{tot_cnt}",
                TEXT_HEAD,
            ),
            (x + 10, y + int(28 * app.scale)),
        )
        app.screen.blit(app.text_cache.render(app.f_tiny, f"Mapa: {scen_name}", TEXT_MUTED), (x + 10, y + int(48 * app.scale)))
    else:
        app.screen.blit(app.text_cache.render(app.f_sub, "⚪ SIMULÁTOR ZASTAVEN", TEXT_MUTED), (x + 10, y + 8))
        app.screen.blit(app.text_cache.render(app.f_small, "Připraven ke startu nové kampaně.", TEXT_BODY), (x + 10, y + int(28 * app.scale)))
        app.screen.blit(app.text_cache.render(app.f_tiny, app.sim_status_line[:40], ACCENT_AMBER), (x + 10, y + int(48 * app.scale)))

    y += box_h + 12

    # Tlačítka start/náhled
    btn_h = max(38, int(42 * app.scale))
    if app.sim_active:
        app.draw_button("open_preview", pygame.Rect(x, y, w, btn_h), "👁️ ŽIVÝ NÁHLED BOJIŠTĚ", "60 FPS vykreslení (Mezerník)", active=True, accent=ACCENT_CYAN)
        y += btn_h + 8

        half_w = (w - 8) // 2
        p_lbl = "▶ POKRAČOVAT" if app.sim_paused else "⏸ POZASTAVIT"
        app.draw_button("toggle_pause", pygame.Rect(x, y, half_w, 32), p_lbl, active=True, accent=ACCENT_GREEN if app.sim_paused else ACCENT_AMBER)
        app.draw_button("stop_sim", pygame.Rect(x + half_w + 8, y, half_w, 32), "⏹ UKONČIT", active=True, accent=ACCENT_ROSE)
        y += 40

        app.screen.blit(app.text_cache.render(app.f_tiny, "RYCHLOST VÝPOČTU:", TEXT_MUTED), (x, y))
        y += 16
        sp_w = (w - 20) // 6
        for idx, sp_lbl in enumerate(app.speed_presets):
            app.draw_button(
                f"sim_speed_{idx}",
                pygame.Rect(x + idx * (sp_w + 4), y, sp_w, 24),
                sp_lbl,
                active=(app.sim_speed_idx == idx),
                accent=ACCENT_AMBER if idx >= 4 else ACCENT_CYAN,
            )
        y += 32
    else:
        s_tag = "Pokračovat z vybraného modelu" if app.start_from_snapshot else "Start nové čisté evoluce"
        app.draw_button("start_continuous", pygame.Rect(x, y, w, btn_h), "▶ SPUSTIT TRÉNOVAT", s_tag, active=True, accent=ACCENT_GREEN)
        y += btn_h + 8

        bw = (w - 8) // 3
        for idx, ep_cnt in enumerate([10, 25, 50]):
            app.draw_button(f"start_batch_{ep_cnt}", pygame.Rect(x + idx * (bw + 4), y, bw, 28), f"+{ep_cnt} EPOCH", active=True, accent=ACCENT_AMBER)
        y += 36

        half_w = (w - 8) // 2
        app.draw_button("mode_new", pygame.Rect(x, y, half_w, 26), "🌱 ČISTÁ KAMPAŇ", active=(not app.start_from_snapshot), accent=ACCENT_GREEN)
        app.draw_button("mode_snap", pygame.Rect(x + half_w + 8, y, half_w, 26), "📂 ZE SNAPSHOTU", active=app.start_from_snapshot, accent=ACCENT_CYAN)
        y += 34

    # Sekce 2: Parametry bojiště
    t_sec2 = app.text_cache.render(app.f_title, "2. PARAMETRY BOJE", ACCENT_AMBER)
    app.screen.blit(t_sec2, (x, y))
    y += int(24 * app.scale)

    mw = (w - 8) // 3
    for s_idx, m_name in enumerate(["DOMINATION", "CITADEL", "CHASM"]):
        app.draw_button(f"scen_{s_idx}", pygame.Rect(x + s_idx * (mw + 4), y, mw, 26), m_name, active=(app.selected_scenario == s_idx), accent=ACCENT_CYAN, disabled=app.sim_active)
    y += 32

    uw = (w - 12) // 4
    b_labels = ["150 b.", "250 b.", "350 b.", "500 b."]
    for u_idx, b_lbl in enumerate(b_labels):
        app.draw_button(f"units_{u_idx}", pygame.Rect(x + u_idx * (uw + 4), y, uw, 24), b_lbl, active=(app.selected_units_idx == u_idx), accent=ACCENT_AMBER, disabled=app.sim_active)
    y += 32

    # Sekce 3: Uložené modely (.pt)
    t_sec3 = app.text_cache.render(app.f_title, "3. ARCHIV MODELŮ (.PT)", ACCENT_CYAN)
    app.screen.blit(t_sec3, (x, y))
    y += int(24 * app.scale)

    list_h = rect.bottom - y - 46
    list_rect = pygame.Rect(x, y, w, max(50, list_h))
    app.draw_card(list_rect, bg=CARD_BG, border=BORDER_SUBTLE, radius=6)

    app.snapshot_box_rect = list_rect
    if not app.snapshots:
        app.screen.blit(app.text_cache.render(app.f_small, "Žádné uložené modely.", TEXT_MUTED), (x + 12, y + 14))
    else:
        row_step = max(20, int(22 * app.scale))
        max_rows = max(1, (list_rect.height - 8) // row_step)
        max_scroll = max(0, len(app.snapshots) - max_rows)
        app.snapshot_scroll_offset = max(0, min(app.snapshot_scroll_offset, max_scroll))
        visible_snaps = app.snapshots[app.snapshot_scroll_offset : app.snapshot_scroll_offset + max_rows]

        for row_i, sp in enumerate(visible_snaps):
            idx = app.snapshot_scroll_offset + row_i
            ry = list_rect.y + 4 + row_i * row_step
            r_rect = pygame.Rect(x + 4, ry, w - 8, row_step - 2)
            app.snapshot_rects.append((idx, r_rect))
            is_sel = idx == app.selected_snapshot_idx
            if is_sel:
                pygame.draw.rect(app.screen, (24, 46, 68), r_rect, border_radius=4)
                pygame.draw.rect(app.screen, ACCENT_CYAN, r_rect, width=1, border_radius=4)
            sz_mb = os.path.getsize(sp) / (1024 ** 2) if os.path.exists(sp) else 0.0
            bname = os.path.basename(sp)
            s_lbl = f"{'' if is_sel else '  '}{bname[:28]} ({sz_mb:.1f}MB)"
            app.screen.blit(app.text_cache.render(app.f_tiny, s_lbl, ACCENT_CYAN if is_sel else TEXT_BODY), (r_rect.x + 6, r_rect.y + 3))

        if len(app.snapshots) > max_rows:
            scroll_h = max(14, int((max_rows / len(app.snapshots)) * list_rect.height))
            scroll_y = list_rect.y + int((app.snapshot_scroll_offset / float(max_scroll)) * (list_rect.height - scroll_h))
            pygame.draw.rect(app.screen, (50, 70, 95), (list_rect.right - 5, scroll_y, 4, scroll_h), border_radius=2)

    y = list_rect.bottom + 8
    bw = (w - 8) // 3
    app.draw_button("run_lab", pygame.Rect(x, y, bw, 30), "📊 REPORT", accent=ACCENT_GREEN)
    app.draw_button("export_csv", pygame.Rect(x + bw + 4, y, bw, 30), "💾 EXPORT", accent=ACCENT_CYAN)
    app.draw_button("clean_logs", pygame.Rect(x + 2 * (bw + 4), y, bw, 30), "🧹 VYČISTIT", accent=ACCENT_ROSE)


def draw_graphical_heatmap(app: CyberTacticsLauncher, r_box: pygame.Rect) -> None:
    """Vykreslí 2D tepelnou mapu bojových událostí."""
    pygame.draw.rect(app.screen, (7, 10, 15), r_box, border_radius=6)
    rows, cols = app.heatmap_grid.shape
    cw = r_box.width / float(cols)
    ch = r_box.height / float(rows)
    max_v = max(1, int(np.max(app.heatmap_grid)))

    for r in range(rows):
        for c in range(cols):
            v = app.heatmap_grid[r, c]
            px = int(r_box.x + c * cw)
            py = int(r_box.y + r * ch)
            pw = max(1, int(r_box.x + (c + 1) * cw) - px - 1)
            ph = max(1, int(r_box.y + (r + 1) * ch) - py - 1)

            if v == 0:
                col = (13, 17, 26)
            else:
                ratio = min(1.0, math.sqrt(v / float(max_v)))
                if ratio < 0.5:
                    t = ratio * 2.0
                    col = (int(15 + 230 * t), int(40 + 140 * t), int(90 * (1.0 - t) + 20))
                else:
                    t = (ratio - 0.5) * 2.0
                    col = (255, int(180 * (1.0 - t) + 30), int(20 + 80 * t))
            pygame.draw.rect(app.screen, col, (px, py, pw, ph))
    pygame.draw.rect(app.screen, (28, 38, 56), r_box, width=1, border_radius=6)


def draw_modern_radar(app: CyberTacticsLauncher, rect: pygame.Rect, state: dict) -> None:
    """Střední velký taktický Holo-Radar (přepínatelný s heatmapou)."""
    app.draw_card(rect, bg=PANEL_SURFACE, border=BORDER_SUBTLE, radius=8)

    scen_name = state.get("scenario_name", "Bojiště").split("(")[0].strip()
    hdr_str = f"🛰️ TAKTICKÝ RADAR // {scen_name.upper()}"
    app.screen.blit(app.text_cache.render(app.f_title, hdr_str, ACCENT_CYAN), (rect.x + 14, rect.y + 12))

    btn_w = 110
    app.draw_button("tab_radar", pygame.Rect(rect.right - 2 * btn_w - 18, rect.y + 10, btn_w, 28), "RADAR", active=(app.view_mode == "RADAR"), accent=ACCENT_CYAN)
    app.draw_button("tab_heatmap", pygame.Rect(rect.right - btn_w - 14, rect.y + 10, btn_w, 28), "HEATMAPA", active=(app.view_mode == "HEATMAP"), accent=ACCENT_AMBER)

    pad_top = 46
    inner_w = rect.width - 24
    inner_h = rect.height - pad_top - 14

    aspect = 1600.0 / 1000.0
    rw = inner_w
    rh = int(rw / aspect)
    if rh > inner_h:
        rh = inner_h
        rw = int(rh * aspect)

    rx = rect.x + 12 + (inner_w - rw) // 2
    ry = rect.y + pad_top + (inner_h - rh) // 2
    r_box = pygame.Rect(rx, ry, rw, rh)

    if app.view_mode == "RADAR":
        pygame.draw.rect(app.screen, (7, 10, 15), r_box, border_radius=6)
        pygame.draw.rect(app.screen, (28, 38, 56), r_box, width=1, border_radius=6)

        def to_r(wx, wy):
            return rx + int((wx / 1600.0) * rw), ry + int((wy / 1000.0) * rh)

        faction_colors = [m["color"] for m in FACTION_THEMES]
        r_scale = rw / 1600.0

        bf_render.draw_static_map(app.screen, r_box, to_r, r_scale, state, faction_colors, mode="radar", step_px=max(24, rw // 5))
        bf_render.draw_gates(app.screen, state.get("gates", []), to_r, faction_colors, mode="radar")
        bf_render.draw_nodes(app.screen, state.get("nodes", []), to_r, r_scale, faction_colors)
        bf_render.draw_tracers(app.screen, state, to_r, faction_colors)
        bf_render.draw_heals(app.screen, state, to_r)
        bf_render.draw_lasers(app.screen, state, to_r)
        bf_render.draw_smoke(app.screen, state.get("smokes", []), to_r, r_scale)
        bf_render.draw_units(app.screen, state, to_r, r_scale, 1.0, faction_colors, mode="radar", show_hp=False)
    else:
        draw_graphical_heatmap(app, r_box)


def draw_modern_analytics(app: CyberTacticsLauncher, rect: pygame.Rect, state: dict) -> None:
    """Pravý panel: Válečný log, živá ekonomika ROI a křivka evoluce."""
    app.draw_card(rect, bg=PANEL_SURFACE, border=BORDER_SUBTLE, radius=8)

    x = rect.x + 14
    w = rect.width - 28
    y = rect.y + 12

    # 1. Živý válečný protokol
    wire_h = max(90, int(105 * app.scale))
    app.screen.blit(app.text_cache.render(app.f_title, "⚡ ŽIVÝ PROTOKOL BOJE", ACCENT_AMBER), (x, y))
    y += int(22 * app.scale)

    app.draw_card(pygame.Rect(x, y, w, wire_h), bg=CARD_BG, border=BORDER_SUBTLE, radius=6)
    events = state.get("events", [])
    line_step = max(16, int(17 * app.scale))
    max_lines = max(1, (wire_h - 10) // line_step)
    ey = y + 6
    for ev in events[-max_lines:]:
        tm = ev.get("team", -1)
        col = FACTION_THEMES[tm]["color"] if 0 <= tm < 4 else ACCENT_AMBER
        ev_txt = ev.get("text", "")
        if len(ev_txt) > 42:
            ev_txt = ev_txt[:40] + ".."
        app.screen.blit(app.text_cache.render(app.f_small, ev_txt, col), (x + 10, ey))
        ey += line_step
    y += wire_h + 12

    # 2. Ekonomika a ROI tříd
    roi_h = max(110, int(125 * app.scale))
    app.screen.blit(app.text_cache.render(app.f_title, "💰 EKONOMIKA TŘÍD (ROI)", ACCENT_CYAN), (x, y))
    y += int(22 * app.scale)

    app.draw_card(pygame.Rect(x, y, w, roi_h), bg=CARD_BG, border=BORDER_SUBTLE, radius=6)
    th_y = y + 6
    app.screen.blit(app.text_cache.render(app.f_tiny, "TŘÍDA", ACCENT_CYAN), (x + 10, th_y))
    app.screen.blit(app.text_cache.render(app.f_tiny, "CENA", ACCENT_CYAN), (x + int(w * 0.28), th_y))
    app.screen.blit(app.text_cache.render(app.f_tiny, "ŽIVÝCH", ACCENT_CYAN), (x + int(w * 0.44), th_y))
    app.screen.blit(app.text_cache.render(app.f_tiny, "DMG", ACCENT_CYAN), (x + int(w * 0.64), th_y))
    app.screen.blit(app.text_cache.render(app.f_tiny, "ROI / 1b", ACCENT_CYAN), (x + int(w * 0.82), th_y))

    live_stats = state.get("live_class_stats", [])
    row_y = th_y + 16
    for i, cs in enumerate(live_stats[:4]):
        c_col = CLASS_COLORS[i % 4]
        app.screen.blit(app.text_cache.render(app.f_small, cs["name"], c_col), (x + 10, row_y))
        app.screen.blit(app.text_cache.render(app.f_small, f"{cs['cost']} b", TEXT_BODY), (x + int(w * 0.28), row_y))
        app.screen.blit(app.text_cache.render(app.f_small, f"{cs['alive']}/{cs['total']}", TEXT_HEAD), (x + int(w * 0.44), row_y))
        app.screen.blit(app.text_cache.render(app.f_small, f"{cs['damage']:.0f}", TEXT_BODY), (x + int(w * 0.64), row_y))
        app.screen.blit(app.text_cache.render(app.f_small, f"{cs['roi']:.1f}", ACCENT_GREEN), (x + int(w * 0.82), row_y))
        row_y += int(19 * app.scale)
    y += roi_h + 12

    # 3. Křivka učení AI (fitness evolution)
    ev_h = rect.bottom - y - 14
    app.screen.blit(app.text_cache.render(app.f_title, "📈 EVOLUCE FITNESU (ADAPTACE)", ACCENT_GREEN), (x, y))
    y += int(22 * app.scale)

    graph_rect = pygame.Rect(x, y, w, max(40, ev_h - 22))
    app.draw_card(graph_rect, bg=CARD_BG, border=BORDER_SUBTLE, radius=6)

    hist = state.get("history_avg_fitness", [[] for _ in range(4)])
    all_v = [val for f_s in hist for val in list(f_s)[-30:]]
    if not all_v and app.epochs_data:
        for ep in app.epochs_data[-25:]:
            for f_idx in range(4):
                v = ep.get("factions", [{}])[f_idx].get("fitness_avg", 0.0) if f_idx < len(ep.get("factions", [])) else 0.0
                all_v.append(float(v))

    if not all_v:
        app.screen.blit(app.text_cache.render(app.f_small, "Čekám na odehrání generací...", TEXT_MUTED), (graph_rect.x + 14, graph_rect.centery - 8))
    else:
        min_v = min(all_v)
        max_v = max(all_v)
        span = max(10.0, max_v - min_v)
        gw = graph_rect.width - 24
        gh = graph_rect.height - 20
        gx = graph_rect.x + 12
        gy = graph_rect.y + 10

        for f in range(4):
            f_list = list(hist[f]) if f < len(hist) and hist[f] else []
            series = f_list[-30:] if f_list else []
            if not series:
                continue
            col = FACTION_THEMES[f]["color"]
            if len(series) == 1:
                py = int(gy + gh - ((series[0] - min_v) / span) * gh)
                pygame.draw.circle(app.screen, col, (gx + gw // 2, py), 4)
            else:
                pts = []
                for idx, val in enumerate(series):
                    px = int(gx + (idx / float(len(series) - 1)) * gw)
                    py = int(gy + gh - ((val - min_v) / span) * gh)
                    pts.append((px, max(gy, min(gy + gh, py))))
                pygame.draw.lines(app.screen, col, False, pts, 2)
