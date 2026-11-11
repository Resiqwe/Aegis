"""battlefield_renderer.py - Sdílený renderer bojiště pro launcher radar i taktický HUD.

Poskytuje jednotný vzhled mapy ve dvou režimech:
  - mode="radar": zjednodušené, vysoce kontrastní prvky pro malý náhled.
  - mode="full":  plné 3D stínování zdí, šrafování, výrazné siluety jednotek.

Všechny funkce přijímají transformaci `to_screen(x, y) -> (sx, sy)` a
`radius_scale` (pixely na 1 světovou jednotku) pro konzistentní škálování.
"""

import math

import pygame

from ai_brains import CLASS_GRUNT, CLASS_MEDIC, CLASS_SNIPER, CLASS_JUGGERNAUT

WALL_FILL = (40, 53, 74)
WALL_TOP = (60, 79, 106)
WALL_EDGE = (104, 130, 162)
WALL_SHADOW = (17, 23, 33)
WALL_HATCH = (52, 68, 92)
PILLAR_FILL = (34, 46, 66)
PILLAR_EDGE = (92, 118, 150)
GRID_COL = (17, 23, 33)
SMOKE_COL = (74, 90, 82)

_SMOKE_CACHE = {}
_STATIC_CACHE = {}
_STATIC_CACHE_MAX = 8


def draw_grid(surface, rect, step_px=48, color=GRID_COL):
    x = rect.x
    while x < rect.right:
        pygame.draw.line(surface, color, (x, rect.y), (x, rect.bottom), 1)
        x += step_px
    y = rect.y
    while y < rect.bottom:
        pygame.draw.line(surface, color, (rect.x, y), (rect.right, y), 1)
        y += step_px


def draw_static_map(surface, rect, to_screen, radius_scale, state, faction_colors,
                    mode="radar", step_px=48):
    """Vykreslí statické prvky (mřížka, základny, zdi, pilíře) přes cached Surface.

    Cache se invaliduje změnou velikosti rectu, seznamu zdí (zničená brána),
    pilířů, základen nebo režimu. Dynamické prvky (uzly, brány, jednotky,
    efekty) se kreslí zvlášť každý frame.
    """
    walls = state.get("walls", [])
    bases = state.get("bases", [])
    pillars = state.get("pillars", [])
    key = (
        mode, rect.width, rect.height, step_px,
        tuple(tuple(w) for w in walls),
        tuple(tuple(p) for p in pillars),
        tuple((b["x"], b["y"], b["r"], b["team"]) for b in bases),
    )
    cached = _STATIC_CACHE.get(key)
    if cached is None:
        surf = pygame.Surface((max(1, rect.width), max(1, rect.height)), pygame.SRCALPHA)

        def local(x, y):
            sx, sy = to_screen(x, y)
            return sx - rect.x, sy - rect.y

        draw_grid(surf, pygame.Rect(1, 1, max(1, rect.width - 2), max(1, rect.height - 2)),
                  step_px=step_px)
        draw_bases(surf, bases, local, radius_scale, faction_colors)
        draw_walls(surf, walls, local, mode=mode)
        draw_pillars(surf, pillars, local, radius_scale, mode=mode)
        if len(_STATIC_CACHE) >= _STATIC_CACHE_MAX:
            _STATIC_CACHE.clear()
        _STATIC_CACHE[key] = surf
        cached = surf
    surface.blit(cached, rect.topleft)


def draw_walls(surface, walls, to_screen, mode="full"):
    for w in walls:
        x1, y1 = to_screen(w[0], w[1])
        x2, y2 = to_screen(w[2], w[3])
        rw = max(2, x2 - x1)
        rh = max(2, y2 - y1)
        if mode == "full":
            pygame.draw.rect(surface, WALL_SHADOW, (x1 + 2, y1 + 3, rw, rh), border_radius=2)
            pygame.draw.rect(surface, WALL_FILL, (x1, y1, rw, rh), border_radius=2)
            if rw >= 8 and rh >= 8:
                pygame.draw.line(surface, WALL_TOP, (x1 + 1, y1 + 1), (x2 - 1, y1 + 1), 2)
                if rw >= 16 and rh >= 12:
                    step = 9
                    start = x1 - rh
                    while start < x2:
                        sx = max(x1, start)
                        sy = y1 + (sx - start)
                        ex = min(x2, sy - y1 + x1)
                        ey = y1 + (ex - start)
                        if ex > sx and ey > sy:
                            pygame.draw.line(surface, WALL_HATCH, (sx, sy), (ex, ey), 1)
                        start += step
            pygame.draw.rect(surface, WALL_EDGE, (x1, y1, rw, rh), width=1, border_radius=2)
        else:
            pygame.draw.rect(surface, WALL_FILL, (x1, y1, rw, rh))
            pygame.draw.rect(surface, WALL_EDGE, (x1, y1, rw, rh), width=1)


def draw_pillars(surface, pillars, to_screen, radius_scale, mode="full"):
    for p in pillars:
        sx, sy = to_screen(p[0], p[1])
        pr = max(3, int(p[2] * radius_scale))
        pygame.draw.circle(surface, WALL_SHADOW, (sx + 1, sy + 2), pr)
        pygame.draw.circle(surface, PILLAR_FILL, (sx, sy), pr)
        pygame.draw.circle(surface, PILLAR_EDGE, (sx, sy), pr, width=1)
        if mode == "full" and pr >= 8:
            pygame.draw.circle(surface, WALL_TOP, (sx, sy), max(2, pr // 2), width=1)


def draw_bases(surface, bases, to_screen, radius_scale, faction_colors):
    for b in bases:
        sx, sy = to_screen(b["x"], b["y"])
        sr = max(8, int(b["r"] * radius_scale))
        col = faction_colors[b["team"] % len(faction_colors)]
        pygame.draw.circle(surface, (col[0] // 8, col[1] // 8, col[2] // 8), (sx, sy), sr)
        pygame.draw.circle(surface, col, (sx, sy), sr, width=1)


def draw_nodes(surface, nodes, to_screen, radius_scale, faction_colors):
    for n in nodes:
        nx, ny, nr, n_owner, n_cap = n
        sx, sy = to_screen(nx, ny)
        sr = max(5, int(nr * radius_scale))
        owner_id = int(n_owner)
        col = faction_colors[owner_id % len(faction_colors)] if owner_id >= 0 else (90, 105, 125)
        pygame.draw.circle(surface, (col[0] // 6, col[1] // 6, col[2] // 6), (sx, sy), sr)
        pygame.draw.circle(surface, col, (sx, sy), sr, width=2)
        if n_cap > 0.05:
            pygame.draw.circle(surface, col, (sx, sy), max(2, int(sr * n_cap)), width=2)


def _smoke_sprite(radius):
    r = max(6, int(radius))
    key = r
    sprite = _SMOKE_CACHE.get(key)
    if sprite is not None:
        return sprite
    size = r * 2 + 4
    sprite = pygame.Surface((size, size), pygame.SRCALPHA)
    cx = cy = size // 2
    layers = [
        (1.00, 34),
        (0.80, 44),
        (0.58, 56),
        (0.36, 68),
    ]
    for scale, alpha in layers:
        pygame.draw.circle(sprite, (*SMOKE_COL, alpha), (cx, cy), int(r * scale))
    for ang in range(0, 360, 30):
        px = cx + int(math.cos(math.radians(ang)) * r * 0.92)
        py = cy + int(math.sin(math.radians(ang)) * r * 0.92)
        pygame.draw.circle(sprite, (*SMOKE_COL, 90), (px, py), 1)
    _SMOKE_CACHE[key] = sprite
    return sprite


def draw_smoke(surface, smokes, to_screen, radius_scale):
    for s in smokes:
        sx, sy = to_screen(s[0], s[1])
        radius = max(6, int(s[2] * radius_scale))
        life = max(0.0, min(1.0, float(s[3]) / 2.0))
        sprite = _smoke_sprite(radius)
        sprite.set_alpha(int(255 * (0.35 + 0.65 * life)))
        surface.blit(sprite, (sx - sprite.get_width() // 2, sy - sprite.get_height() // 2))


def draw_tracers(surface, state, to_screen, faction_colors):
    t_p1 = state.get("tracers_p1", [])
    t_p2 = state.get("tracers_p2", [])
    t_team = state.get("tracers_team", [])
    for i in range(len(t_p1)):
        x1, y1 = to_screen(t_p1[i, 0], t_p1[i, 1])
        x2, y2 = to_screen(t_p2[i, 0], t_p2[i, 1])
        tm = int(t_team[i]) if i < len(t_team) else 0
        col = faction_colors[tm] if 0 <= tm < len(faction_colors) else (255, 255, 255)
        pygame.draw.line(surface, col, (x1, y1), (x2, y2), 1)


def draw_heals(surface, state, to_screen):
    h_p1 = state.get("heals_p1", [])
    h_p2 = state.get("heals_p2", [])
    for i in range(len(h_p1)):
        x1, y1 = to_screen(h_p1[i, 0], h_p1[i, 1])
        x2, y2 = to_screen(h_p2[i, 0], h_p2[i, 1])
        pygame.draw.line(surface, (60, 250, 130), (x1, y1), (x2, y2), 1)


def draw_lasers(surface, state, to_screen):
    for i, laser in enumerate(state.get("sniper_lasers", [])):
        x1, y1 = to_screen(laser[0], laser[1])
        x2, y2 = to_screen(laser[2], laser[3])
        pygame.draw.line(surface, (255, 50, 70), (x1, y1), (x2, y2), 1)


def draw_sparks(surface, state, to_screen, mode="full"):
    for spark in state.get("sparks", []):
        sx, sy = to_screen(spark[0], spark[1])
        pygame.draw.circle(surface, (255, 235, 130), (sx, sy), 3 if mode == "full" else 1)
    for spark in state.get("armor_sparks", []):
        sx, sy = to_screen(spark[0], spark[1])
        pygame.draw.circle(surface, (100, 220, 255), (sx, sy), 4 if mode == "full" else 2)


def draw_particles(surface, state, to_screen, zoom=1.0):
    for p in state.get("particles", []):
        life = max(0.0, min(1.0, float(p.get("life", 1.0))))
        if life <= 0.05:
            continue
        px, py = to_screen(p["x"], p["y"])
        r_p = max(1, int(3 * life * zoom))
        col = p.get("col", (255, 180, 50))
        c_safe = (
            max(0, min(255, int(col[0] * life))),
            max(0, min(255, int(col[1] * life))),
            max(0, min(255, int(col[2] * life))),
        )
        pygame.draw.circle(surface, c_safe, (px, py), r_p)


def _rot(sx, sy, ang, dist):
    return (sx + math.cos(ang) * dist, sy + math.sin(ang) * dist)


def draw_unit(surface, sx, sy, heading, color, class_id, size, mode="full"):
    """Výrazné unikátní siluety: Pěšák=chevron, Medik=delta+kříž, Sniper=diamant, Juggernaut=šestiúhelník."""
    th = heading
    dark = (max(0, color[0] // 4), max(0, color[1] // 4), max(0, color[2] // 4))

    if class_id == CLASS_JUGGERNAUT:
        r_j = size * 1.55
        pts = [
            (sx + math.cos(th + i * math.pi / 3) * r_j, sy + math.sin(th + i * math.pi / 3) * r_j)
            for i in range(6)
        ]
        pygame.draw.polygon(surface, dark, pts)
        pygame.draw.polygon(surface, color, pts, width=3)
        inner = [(sx + (px - sx) * 0.58, sy + (py - sy) * 0.58) for px, py in pts]
        pygame.draw.polygon(surface, color, inner, width=1)
        pygame.draw.line(surface, (245, 250, 255), (sx, sy), _rot(sx, sy, th, r_j + 2), 2)
        return

    if class_id == CLASS_SNIPER:
        nose = _rot(sx, sy, th, size * 1.9)
        tail = _rot(sx, sy, th + math.pi, size * 1.15)
        left = _rot(sx, sy, th + math.pi / 2, size * 0.62)
        right = _rot(sx, sy, th - math.pi / 2, size * 0.62)
        pygame.draw.polygon(surface, color, [nose, right, tail, left])
        pygame.draw.circle(surface, (255, 255, 255), (int(nose[0]), int(nose[1])), max(1, int(size * 0.22)))
        return

    # Pěšák: ostrý útočný klín (chevron)
    nose = _rot(sx, sy, th, size * 1.45)
    left = _rot(sx, sy, th + 2.35, size * 1.05)
    right = _rot(sx, sy, th - 2.35, size * 1.05)
    back = _rot(sx, sy, th + math.pi, size * 0.35)
    pygame.draw.polygon(surface, color, [nose, left, back, right])
    if class_id == CLASS_MEDIC:
        pygame.draw.line(surface, (80, 255, 130), (sx - size * 0.45, sy), (sx + size * 0.45, sy), 2)
        pygame.draw.line(surface, (80, 255, 130), (sx, sy - size * 0.45), (sx, sy + size * 0.45), 2)
    elif mode == "full":
        pygame.draw.line(surface, (245, 250, 255), (sx, sy), _rot(sx, sy, th, size * 1.2), 1)


def draw_units(surface, state, to_screen, radius_scale, zoom, faction_colors,
               mode="full", tracked_id=-1, show_hp=True):
    pos = state.get("pos", [])
    head = state.get("heading", [])
    hp = state.get("hp", [])
    max_hp = state.get("max_hp", [])
    teams = state.get("teams", [])
    classes = state.get("classes", [])
    alive = state.get("alive", [])
    supp = state.get("suppression", [])
    cover = state.get("in_cover", [])
    reloading = state.get("reloading", [])

    if mode == "full":
        base_size = max(5.5, 9.5 * zoom)
    else:
        base_size = max(2.0, 2.2 * min(2.0, zoom))

    for i in range(len(pos)):
        if not alive[i]:
            continue
        sx, sy = to_screen(pos[i, 0], pos[i, 1])
        tm = int(teams[i]) if i < len(teams) else 0
        col = faction_colors[tm] if 0 <= tm < len(faction_colors) else (255, 255, 255)
        cid = int(classes[i]) if i < len(classes) else CLASS_GRUNT
        th = float(head[i]) if i < len(head) else 0.0

        draw_unit(surface, sx, sy, th, col, cid, base_size, mode)

        # Indikace aktivního kinetického štítu (cyan/modré energetické silové pole)
        has_shield = state.get("has_kinetic_shield")
        if has_shield is not None and i < len(has_shield) and has_shield[i]:
            pygame.draw.circle(surface, (80, 220, 255), (sx, sy), int(base_size) + (4 if mode == "full" else 2), width=1)

        if mode == "full":
            if i < len(cover) and cover[i]:
                pygame.draw.circle(surface, (100, 160, 240), (sx, sy), int(base_size) + 3, width=1)
            if i < len(supp) and supp[i] > 0.35:
                pygame.draw.circle(surface, (255, 60, 60), (sx, sy), int(base_size) + 4, width=1)
            if show_hp and i < len(hp):
                frac = max(0.0, min(1.0, hp[i] / max(1.0, max_hp[i])))
                bw = max(10, int(14 * zoom))
                by = sy - int(base_size) - 6
                pygame.draw.rect(surface, (32, 16, 16), (sx - bw // 2, by, bw, 3))
                hp_col = (50, 240, 80) if frac > 0.4 else (240, 50, 50)
                pygame.draw.rect(surface, hp_col, (sx - bw // 2, by, int(bw * frac), 3))
            if i < len(reloading) and reloading[i] and zoom > 1.3:
                pygame.draw.circle(
                    surface, (255, 190, 60),
                    (sx + int(base_size) + 4, sy - int(base_size) - 4), 2,
                )

        if i == tracked_id:
            pygame.draw.circle(surface, (255, 255, 255), (sx, sy), int(base_size) + 7, width=1)


def draw_gates(surface, gates, to_screen, faction_colors, mode="full"):
    for g in gates:
        x1, y1, x2, y2 = g["rect"]
        sx1, sy1 = to_screen(x1, y1)
        sx2, sy2 = to_screen(x2, y2)
        rw = max(2, sx2 - sx1)
        rh = max(2, sy2 - sy1)
        tcol = faction_colors[int(g.get("team", 0)) % len(faction_colors)]
        if g.get("destroyed", False):
            for dx in range(0, rw, 6):
                for dy in range(0, rh, 6):
                    pygame.draw.circle(surface, (60, 64, 70), (sx1 + dx, sy1 + dy), 1)
            continue
        pygame.draw.rect(surface, (58, 70, 84), (sx1, sy1, rw, rh), border_radius=2)
        pygame.draw.rect(surface, tcol, (sx1, sy1, rw, rh), width=2, border_radius=2)
        if mode != "full":
            continue
        frac = max(0.0, min(1.0, float(g.get("hp", 1.0)) / max(1.0, float(g.get("max_hp", 1.0)))))
        if rh >= 6:
            bar_y = sy1 - 4
            pygame.draw.rect(surface, (30, 16, 18), (sx1, bar_y, rw, 3))
            bar_col = (70, 220, 110) if frac > 0.5 else ((240, 190, 50) if frac > 0.2 else (240, 60, 60))
            pygame.draw.rect(surface, bar_col, (sx1, bar_y, int(rw * frac), 3))
