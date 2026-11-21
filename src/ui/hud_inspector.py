"""
ui/hud_inspector.py - Pravý inspekční dock pro OptimizedTacticalHUD.

Rozdělený na 4 samostatné sekce:
1. Frakce, generálové (ROI), techy a doktríny
2. Křivka učení AI (fitness) a bojový protokol událostí
3. Rentgen hierarchického mozku vybraného vojáka (14 -> H1 24 -> H2 16 -> 7)
4. Živá ekonomika tříd (ROI) a strategické body uzlů
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pygame

from src.ai_brains import CLASS_MEDIC
from src.engine_core import DOCTRINE_LIST
from src.ui.widgets import (
    ACCENT_AMBER,
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_PINK,
    BORDER_COL,
    CLASS_COLORS,
    FACTION_META,
    PANEL_BG,
    PANEL_HEADER,
    TEXT_DIM,
    TEXT_MAIN,
)

if TYPE_CHECKING:
    from src.tactical_hud import OptimizedTacticalHUD

_DOC_COLORS = [
    (140, 155, 175),   # BALANCED (neutrální ocelová)
    (255, 75, 75),     # RUSH (agresivní červená)
    (60, 160, 255),    # HOLD (obranná modrá)
    (200, 100, 255),   # SKIRMISH (taktická fialová)
]

_TECH_TAG_MAP = {0: "NANIT", 1: "RADAR", 2: "ŠTÍT", 3: "OVERCLOCK", 4: "AEROSOL"}

_SHORT_SENSORS = [
    "1.Cíl Vpřed", "2.Cíl Vpravo", "3.Blízkost",
    "4.Palba Vpř.", "5.Palba Vpr.",
    "6.Povel Vpř.", "7.Povel Vpr.",
    "8.Potlačení", "9.Zásobník", "10.Zdraví(HP)",
    "11.Kryt Vpř.", "12.Kryt Vpr.",
    "13.Koheze čety", "14.Stav zóny",
]

_SHORT_ACTIONS = [
    "1.Surge(Plyn)", "2.Sway(Úkrok)", "3.Yaw(Míření)", "4.Trigger(Palba)",
    "5.Stance(Sprint)", "6.Ability(Doved.)", "7.Ping(Četa)",
]


def _draw_factions_section(hud: OptimizedTacticalHUD, x: int, y: int, w: int, hdr_h: int, state: dict) -> int:
    pygame.draw.rect(hud.screen, PANEL_HEADER, (x, y, w, hdr_h), border_radius=3)
    hud.screen.blit(
        hud.text_cache.render(hud.f_sub, "1. FRAKCE, GENERÁLOVÉ (ROI) & DOKTRÍNY", ACCENT_CYAN),
        (x + 8, y + max(2, (hdr_h - hud.f_sub.get_height()) // 2)),
    )
    y += hdr_h + 3

    alive_counts = state.get("alive_counts", [0, 0, 0, 0])
    kills = state.get("kills_by_faction", getattr(hud.engine, "epoch_wins_per_faction", [0, 0, 0, 0]))
    wins = getattr(hud.engine, "epoch_wins_per_faction", [0, 0, 0, 0])
    total_alive = max(1, sum(alive_counts))

    # Tug-of-war barevný pruh poměru sil
    bar_h = 6
    cur_bx = x
    for f in range(hud.engine.n_factions):
        seg_w = int((alive_counts[f] / total_alive) * w)
        if seg_w > 0:
            pygame.draw.rect(hud.screen, FACTION_META[f]["color"], (cur_bx, y, seg_w, bar_h))
            cur_bx += seg_w
    pygame.draw.rect(hud.screen, BORDER_COL, (x, y, w, bar_h), width=1)
    y += 9

    generals_info = state.get("generals", [])
    f_techs = state.get("faction_techs", [[] for _ in range(4)])

    for f in range(hud.engine.n_factions):
        f_col = FACTION_META[f]["color"]
        f_alive = alive_counts[f]
        init_c = max(1, hud.engine.initial_team_counts[f] if f < len(hud.engine.initial_team_counts) else 100)
        eff_pct = int((f_alive / init_c) * 100)

        gen = generals_info[f] if f < len(generals_info) else {}
        lvl = gen.get("level", 0)
        comp = gen.get("composition", [100, 0, 0, 0])
        probs = gen.get("probs", [1.0, 0.0, 0.0, 0.0])

        row_h = max(56, int(58 * hud.ui_scale))
        row = pygame.Rect(x, y, w, row_h)
        pygame.draw.rect(hud.screen, (20, 26, 38), row, border_radius=4)
        pygame.draw.rect(hud.screen, BORDER_COL, row, width=1, border_radius=4)
        pygame.draw.rect(hud.screen, f_col, (x, y, 4, row_h), border_radius=2)

        doc_btn_w = max(74, int(80 * hud.ui_scale))
        doc_btn_h = max(26, int(28 * hud.ui_scale))
        doc_btn_y = y + (row_h - doc_btn_h) // 2
        cur_doc = hud.engine.faction_doctrines[f]
        hud.draw_button(
            hud.screen,
            f"doc_{f}",
            pygame.Rect(x + w - doc_btn_w - 6, doc_btn_y, doc_btn_w, doc_btn_h),
            DOCTRINE_LIST[cur_doc],
            active=True,
            accent=_DOC_COLORS[cur_doc],
        )

        k_val = kills[f] if f < len(kills) else 0
        w_val = wins[f] if f < len(wins) else 0
        t_line1 = f"{FACTION_META[f]['short']} L{lvl} W:{w_val} | Síla: {eff_pct}% ({f_alive}) | K: {k_val}"
        hud.screen.blit(
            hud.text_cache.render(hud.f_small, t_line1, f_col if eff_pct > 22 else (230, 50, 50)),
            (x + 12, y + 6),
        )

        tech_tags = [_TECH_TAG_MAP.get(tid, f"T{tid}") for tid in f_techs[f]] if f < len(f_techs) else []
        tech_str = (" [" + "][".join(tech_tags) + "]") if tech_tags else " [BEZ TECH]"

        comp_txt = (
            f"P:{comp[0]} M:{comp[1]} S:{comp[2]} J:{comp[3]}  •  "
            f"Váhy: {int(probs[0]*100)}/{int(probs[1]*100)}/{int(probs[2]*100)}/{int(probs[3]*100)}%"
        )
        hud.screen.blit(hud.text_cache.render(hud.f_tiny, comp_txt, TEXT_DIM), (x + 12, y + 24))
        hud.screen.blit(
            hud.text_cache.render(hud.f_tiny, f"TECH:{tech_str}", ACCENT_AMBER if tech_tags else TEXT_DIM),
            (x + 12, y + 38),
        )
        y += row_h + 4

    return y + 3


def _draw_fitness_and_feed_section(hud: OptimizedTacticalHUD, x: int, y: int, w: int, hdr_h: int, state: dict) -> int:
    pygame.draw.rect(hud.screen, PANEL_HEADER, (x, y, w, hdr_h), border_radius=3)
    hud.screen.blit(
        hud.text_cache.render(hud.f_sub, "2. VÝVOJ UČENÍ (FITNESS) & PROTOKOL", ACCENT_CYAN),
        (x + 8, y + max(2, (hdr_h - hud.f_sub.get_height()) // 2)),
    )
    y += hdr_h + 3

    mid_h = max(92, int(96 * hud.ui_scale))
    graph_w = int(w * 0.46)
    feed_w = w - graph_w - 6

    hud._draw_fitness_graph(pygame.Rect(x, y, graph_w, mid_h), state)

    feed_rect = pygame.Rect(x + graph_w + 6, y, feed_w, mid_h)
    pygame.draw.rect(hud.screen, (13, 17, 25), feed_rect, border_radius=3)
    pygame.draw.rect(hud.screen, BORDER_COL, feed_rect, width=1, border_radius=3)

    events = state.get("events", [])
    ey = y + 4
    ev_step = max(16, int(17 * hud.ui_scale))
    for ev in events[-5:]:
        e_col = FACTION_META[ev["team"]]["color"] if 0 <= ev["team"] < 4 else ACCENT_AMBER
        ev_txt = ev["text"]
        if len(ev_txt) > 28:
            ev_txt = ev_txt[:26] + ".."
        hud.screen.blit(hud.text_cache.render(hud.f_tiny, ev_txt, e_col), (feed_rect.x + 6, ey))
        ey += ev_step

    return y + mid_h + 8


def _draw_brain_xray_section(
    hud: OptimizedTacticalHUD,
    x: int,
    y: int,
    w: int,
    hdr_h: int,
    unit_info: dict | None,
) -> int | None:
    pygame.draw.rect(hud.screen, PANEL_HEADER, (x, y, w, hdr_h), border_radius=3)
    lbl_mode = "MANUÁL" if hud.selected_unit >= 0 else "AUTO-ELITA"
    hud.screen.blit(
        hud.text_cache.render(hud.f_sub, f"3. HIERARCHICKÝ MOZEK (14->24->16->7) // {lbl_mode}", ACCENT_CYAN),
        (x + 8, y + max(2, (hdr_h - hud.f_sub.get_height()) // 2)),
    )
    y += hdr_h + 3

    sens_step = max(13, int(14 * hud.ui_scale))
    insp_h = max(260, int(48 + 14 * sens_step + 12))
    insp_rect = pygame.Rect(x, y, w, insp_h)
    pygame.draw.rect(hud.screen, (13, 17, 25), insp_rect, border_radius=4)
    pygame.draw.rect(hud.screen, BORDER_COL, insp_rect, width=1, border_radius=4)

    if unit_info is None:
        msg = hud.text_cache.render(hud.f_reg, "Žádná jednotka není aktivní.", TEXT_DIM)
        hud.screen.blit(msg, (x + 14, y + 30))
        return None

    f_id = unit_info["team"]
    f_col = FACTION_META[f_id]["color"]
    u_title = (
        f"VOJÁK #{unit_info['unit_idx']:03d} (Slot #{unit_info.get('brain_slot', 0):03d}) "
        f"[{FACTION_META[f_id]['short']}] • {unit_info['class_name']}"
    )
    hud.screen.blit(hud.text_cache.render(hud.f_small, u_title, f_col), (x + 8, y + 4))

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
    hud.screen.blit(hud.text_cache.render(hud.f_tiny, u_stats1, stat_color), (x + 8, y + 19))

    u_stats2 = f"Kills:{unit_info.get('kills', 0)} | Dmg:{dmg_dealt:.0f} | Utrž:{dmg_taken:.0f}{heal_str}"
    hud.screen.blit(hud.text_cache.render(hud.f_tiny, u_stats2, TEXT_DIM), (x + 8, y + 33))

    df = unit_info["decision_flow"]
    col_w = (w - 14) // 2
    col_y_start = y + 48
    bar_w = max(42, int(48 * hud.ui_scale))

    # Levý sloupec: 14 vstupních senzorů
    hud.screen.blit(hud.text_cache.render(hud.f_small, "VSTUPNÍ SENZORY (14)", ACCENT_CYAN), (x + 6, col_y_start))
    for i, (name, val) in enumerate(df["sensors"][:14]):
        ry = col_y_start + 16 + i * sens_step
        s_name = _SHORT_SENSORS[i] if i < len(_SHORT_SENSORS) else name[:12]
        hud.screen.blit(hud.text_cache.render(hud.f_tiny, s_name, TEXT_DIM), (x + 6, ry))

        bx = x + col_w - bar_w - 4
        pygame.draw.rect(hud.screen, (24, 32, 46), (bx, ry + 1, bar_w, 8), border_radius=2)
        half_b = bar_w // 2
        w_b = int(min(1.0, abs(val)) * (half_b - 1))
        c_b = ACCENT_CYAN if val >= 0 else ACCENT_PINK
        if val >= 0:
            pygame.draw.rect(hud.screen, c_b, (bx + half_b, ry + 1, w_b, 8), border_radius=2)
        else:
            pygame.draw.rect(hud.screen, c_b, (bx + half_b - w_b, ry + 1, w_b, 8), border_radius=2)

    # Pravý sloupec: 7 výstupních akcí + H1/H2 skryté vrstvy
    rx = x + col_w + 10
    hud.screen.blit(hud.text_cache.render(hud.f_small, "VÝSTUPNÍ AKCE (7)", ACCENT_GREEN), (rx + 4, col_y_start))
    act_step_7 = max(12, int(13 * hud.ui_scale))
    for i, (name, val) in enumerate(df["actions"][:7]):
        ry = col_y_start + 15 + i * act_step_7
        a_name = _SHORT_ACTIONS[i] if i < len(_SHORT_ACTIONS) else name[:12]
        hud.screen.blit(
            hud.text_cache.render(hud.f_tiny, a_name, TEXT_MAIN if abs(val) > 0.20 else TEXT_DIM),
            (rx + 4, ry),
        )
        bx = rx + col_w - bar_w - 4
        pygame.draw.rect(hud.screen, (24, 32, 46), (bx, ry + 1, bar_w, 7), border_radius=2)
        half_b = bar_w // 2
        w_b = int(min(1.0, abs(val)) * (half_b - 1))
        c_b = ACCENT_GREEN if val >= 0 else ACCENT_AMBER
        if val >= 0:
            pygame.draw.rect(hud.screen, c_b, (bx + half_b, ry + 1, w_b, 7), border_radius=2)
        else:
            pygame.draw.rect(hud.screen, c_b, (bx + half_b - w_b, ry + 1, w_b, 7), border_radius=2)

    hy = col_y_start + 15 + 7 * act_step_7 + 8
    hud.screen.blit(
        hud.text_cache.render(hud.f_tiny, "H1: PERCEPCE PROSTORU (24 NEURONŮ // 2x12)", ACCENT_CYAN),
        (rx + 4, hy),
    )
    h1_vals = df.get("hidden_h1", df.get("hidden", [])[:24])
    n_cols = 12
    cell_gap = 2
    cell_w = max(6, (col_w - (n_cols - 1) * cell_gap - 8) // n_cols)
    cell_h = max(6, int(6.5 * hud.ui_scale))

    for idx, h_val in enumerate(h1_vals[:24]):
        r_idx = idx // n_cols
        c_idx = idx % n_cols
        cx = rx + 4 + c_idx * (cell_w + cell_gap)
        cy = hy + 14 + r_idx * (cell_h + cell_gap)
        val_clamped = 0.0 if (math.isnan(h_val) or math.isinf(h_val)) else max(0.0, min(1.0, abs(h_val)))
        inten = int(val_clamped * 200)
        col_h = (20, min(255, 45 + inten), 200) if h_val >= 0 else (min(255, 50 + inten), 25, 90)
        pygame.draw.rect(hud.screen, col_h, (cx, cy, cell_w, cell_h), border_radius=1)
        pygame.draw.rect(hud.screen, BORDER_COL, (cx, cy, cell_w, cell_h), width=1, border_radius=1)

    h1_rows = (min(len(h1_vals), 24) + n_cols - 1) // n_cols if h1_vals else 2
    h2_y = hy + 14 + max(2, h1_rows) * (cell_h + cell_gap) + 6
    hud.screen.blit(
        hud.text_cache.render(hud.f_tiny, "H2: TAKTICKÁ SYNTÉZA (16 NEURONŮ // 2x8)", ACCENT_AMBER),
        (rx + 4, h2_y),
    )
    h2_vals = df.get("hidden_h2", df.get("hidden", [])[24:40])
    n_cols_h2 = 8
    cell_w_h2 = max(6, (col_w - (n_cols_h2 - 1) * cell_gap - 8) // n_cols_h2)
    for idx, h_val in enumerate(h2_vals[:16]):
        r_idx = idx // n_cols_h2
        c_idx = idx % n_cols_h2
        cx = rx + 4 + c_idx * (cell_w_h2 + cell_gap)
        cy = h2_y + 14 + r_idx * (cell_h + cell_gap)
        val_clamped = 0.0 if (math.isnan(h_val) or math.isinf(h_val)) else max(0.0, min(1.0, abs(h_val)))
        inten = int(val_clamped * 200)
        col_h = (
            (min(255, 30 + inten), min(255, 140 + inten // 2), 255)
            if h_val >= 0
            else (min(255, 180 + inten // 3), 40, 70)
        )
        pygame.draw.rect(hud.screen, col_h, (cx, cy, cell_w_h2, cell_h), border_radius=1)
        pygame.draw.rect(hud.screen, BORDER_COL, (cx, cy, cell_w_h2, cell_h), width=1, border_radius=1)

    return y + insp_h + 8


def _draw_roi_economy_section(hud: OptimizedTacticalHUD, rect: pygame.Rect, x: int, y: int, w: int, state: dict) -> None:
    sec4_h = rect.bottom - y - 4
    if sec4_h < 110:
        return

    pygame.draw.rect(hud.screen, PANEL_HEADER, (x, y, w, 22), border_radius=3)
    hud.screen.blit(hud.text_cache.render(hud.f_sub, "4. EKONOMIKA TŘÍD (ROI) & STATISTIKY BOJE", ACCENT_CYAN), (x + 8, y + 4))
    y += 24

    sec4_rect = pygame.Rect(x, y, w, sec4_h - 24)
    pygame.draw.rect(hud.screen, (13, 17, 25), sec4_rect, border_radius=4)
    pygame.draw.rect(hud.screen, BORDER_COL, sec4_rect, width=1, border_radius=4)

    th_y = y + 5
    hud.screen.blit(
        hud.text_cache.render(hud.f_mono, "TŘÍDA       CENA   ŽIVÝCH    UDĚLENO  UTRŽENO  KILLS   ROI/1b", ACCENT_AMBER),
        (x + 10, th_y),
    )

    stats = state.get("live_class_stats", [])
    for i, cs in enumerate(stats):
        ry = th_y + 18 + i * 17
        c_col = CLASS_COLORS[i % len(CLASS_COLORS)]
        extra_txt = f" +{cs['healing']:.0f}H" if (cs["class_id"] == CLASS_MEDIC and cs["healing"] > 0) else ""
        row_txt = (
            f"{cs['name']:<11} {cs['cost']} b   {cs['alive']:2d}/{cs['total']:2d}    "
            f"{cs['damage']:6.0f}   {cs['damage_taken']:6.0f}    {cs['kills']:2d}    {cs['roi']:5.1f}{extra_txt}"
        )
        hud.screen.blit(hud.text_cache.render(hud.f_mono, row_txt, c_col), (x + 10, ry))

    node_pts = state.get("node_points", [0.0, 0.0, 0.0, 0.0])
    bottom_y = sec4_rect.bottom - 17
    if bottom_y > th_y + 16 + 4 * 17 + 4:
        pts_str = f"Uzly (body): ALPHA:{node_pts[0]:.0f} | OMEGA:{node_pts[1]:.0f} | SOLAR:{node_pts[2]:.0f} | PHANTOM:{node_pts[3]:.0f}"
        hud.screen.blit(hud.text_cache.render(hud.f_tiny, pts_str, TEXT_DIM), (x + 8, bottom_y))


def draw_right_dock(hud: OptimizedTacticalHUD, rect: pygame.Rect, unit_info: dict | None, state: dict) -> None:
    """Vykreslí kompletní pravý inspekční dock HUDu."""
    pygame.draw.rect(hud.screen, PANEL_BG, rect, border_radius=5)
    pygame.draw.rect(hud.screen, BORDER_COL, rect, width=1, border_radius=5)

    pad = 8
    x = rect.x + pad
    w = rect.width - 2 * pad
    y = rect.y + pad
    hdr_h = max(22, int(22 * hud.ui_scale))

    y = _draw_factions_section(hud, x, y, w, hdr_h, state)
    y = _draw_fitness_and_feed_section(hud, x, y, w, hdr_h, state)
    next_y = _draw_brain_xray_section(hud, x, y, w, hdr_h, unit_info)
    if next_y is not None:
        _draw_roi_economy_section(hud, rect, x, next_y, w, state)
