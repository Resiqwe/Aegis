"""
ui/widgets.py - Sdílená barevná paleta, ModernTextCache a základní UI widgety
pro CyberTacticsLauncher i OptimizedTacticalHUD.
"""

from __future__ import annotations

import time
import pygame

# Barevná paleta - Sleek Modern Dark (Launcher)
BG_VOID = (9, 11, 16)
PANEL_SURFACE = (15, 19, 28)
PANEL_ELEVATED = (20, 26, 38)
CARD_BG = (17, 22, 33)
BORDER_SUBTLE = (30, 40, 58)
BORDER_FOCUS = (0, 240, 210)

TEXT_HEAD = (242, 246, 252)
TEXT_BODY = (185, 198, 218)
TEXT_MUTED = (120, 136, 162)

ACCENT_CYAN = (0, 240, 210)
ACCENT_AMBER = (255, 195, 30)
ACCENT_GREEN = (65, 235, 100)
ACCENT_ROSE = (255, 55, 115)
ACCENT_PURPLE = (175, 80, 250)

# Barevná paleta - Tactical HUD
BG_DARK = (11, 14, 21)
PANEL_BG = (16, 21, 31)
PANEL_HEADER = (24, 32, 46)
BORDER_COL = (42, 56, 80)
ACCENT_PINK = (255, 45, 120)
TEXT_MAIN = (235, 242, 250)
TEXT_DIM = (145, 160, 185)

FACTION_THEMES = [
    {"name": "ALPHA VANGUARD", "short": "ALPHA", "color": ACCENT_CYAN, "bg": (0, 35, 32)},
    {"name": "OMEGA SYNDICATE", "short": "OMEGA", "color": ACCENT_ROSE, "bg": (40, 10, 20)},
    {"name": "SOLAR CORE", "short": "SOLAR", "color": ACCENT_AMBER, "bg": (38, 28, 5)},
    {"name": "PHANTOM UNIT", "short": "PHANTOM", "color": ACCENT_PURPLE, "bg": (28, 12, 42)},
]

FACTION_META = [
    {"name": "ALPHA VANGUARD", "short": "ALPHA", "color": (0, 240, 210)},
    {"name": "OMEGA SYNDICATE", "short": "OMEGA", "color": (255, 45, 115)},
    {"name": "SOLAR CORE", "short": "SOLAR", "color": (255, 195, 25)},
    {"name": "PHANTOM UNIT", "short": "PHANTOM", "color": (175, 80, 250)},
]

CLASS_COLORS = [(74, 144, 226), (65, 235, 100), (255, 195, 25), (231, 76, 60)]


def safe_rgb(col, default: tuple[int, int, int] = (200, 200, 200)) -> tuple[int, int, int]:
    """Bezpečně převede a ořízne barvu do rozsahu RGB (0..255)."""
    try:
        return (
            max(0, min(255, int(col[0]))),
            max(0, min(255, int(col[1]))),
            max(0, min(255, int(col[2]))),
        )
    except Exception:
        return default


class ModernTextCache:
    """Blesková textová cache: eliminuje ~95 % CPU overheadu z font.render v každém snímku."""

    def __init__(self, max_entries: int = 2500):
        self._cache: dict[tuple[int, str, tuple], pygame.Surface] = {}
        self._max_entries = max_entries

    def render(self, font: pygame.font.Font, text: str, color: tuple, bold: bool = False) -> pygame.Surface:
        key = (id(font), text, color)
        surf = self._cache.get(key)
        if surf is None:
            surf = font.render(text, True, color)
            if len(self._cache) > self._max_entries:
                self._cache.clear()
            self._cache[key] = surf
        return surf


def draw_card(
    surf: pygame.Surface,
    rect: pygame.Rect,
    bg: tuple[int, int, int] = PANEL_SURFACE,
    border: tuple[int, int, int] | None = BORDER_SUBTLE,
    radius: int = 8,
) -> None:
    """Vykreslí zaoblenou kartu s volitelným okrajem."""
    pygame.draw.rect(surf, bg, rect, border_radius=radius)
    if border:
        pygame.draw.rect(surf, border, rect, width=1, border_radius=radius)


def draw_launcher_button(
    surf: pygame.Surface,
    ui_buttons: dict[str, pygame.Rect],
    text_cache: ModernTextCache,
    f_sub: pygame.font.Font,
    f_tiny: pygame.font.Font,
    key: str,
    rect: pygame.Rect,
    title: str,
    subtitle: str = "",
    active: bool = False,
    accent: tuple[int, int, int] = ACCENT_CYAN,
    radius: int = 6,
    disabled: bool = False,
) -> None:
    """Vykreslí tlačítko v launcheru (s volitelným podtitulkem a hover efektem)."""
    ui_buttons[key] = rect
    mx, my = pygame.mouse.get_pos()
    hover = rect.collidepoint(mx, my) and not disabled

    acc = safe_rgb(accent)
    if disabled:
        bg = (14, 18, 26)
        border = (32, 40, 52)
        text_col = (70, 85, 105)
    elif active:
        bg = (int(acc[0] * 0.18), int(acc[1] * 0.18), int(acc[2] * 0.18))
        border = acc
        text_col = acc
    elif hover:
        bg = PANEL_ELEVATED
        border = (120, 160, 210)
        text_col = TEXT_HEAD
    else:
        bg = CARD_BG
        border = BORDER_SUBTLE
        text_col = TEXT_BODY

    pygame.draw.rect(surf, bg, rect, border_radius=radius)
    pygame.draw.rect(surf, border, rect, width=1, border_radius=radius)

    if subtitle:
        t1 = text_cache.render(f_sub, title, text_col)
        t2 = text_cache.render(f_tiny, subtitle, TEXT_MUTED)
        surf.blit(t1, (rect.x + 12, rect.y + 6))
        surf.blit(t2, (rect.x + 12, rect.y + 24))
    else:
        t1 = text_cache.render(f_sub, title, text_col)
        surf.blit(t1, (rect.centerx - t1.get_width() // 2, rect.centery - t1.get_height() // 2))


def draw_hud_button(
    surf: pygame.Surface,
    ui_buttons: dict[str, pygame.Rect],
    btn_flash: dict[str, float],
    text_cache: ModernTextCache,
    f_small: pygame.font.Font,
    key: str,
    rect: pygame.Rect,
    label: str,
    active: bool = False,
    accent: tuple[int, int, int] = ACCENT_CYAN,
) -> None:
    """Vykreslí tlačítko v Tactical HUDu (včetně 140ms flash efektu při kliknutí)."""
    ui_buttons[key] = rect
    mx, my = pygame.mouse.get_pos()
    hover = rect.collidepoint(mx, my)

    acc = safe_rgb(accent)
    is_flash = (time.perf_counter() - btn_flash.get(key, 0.0)) < 0.14

    if is_flash:
        bg = (
            min(255, int(acc[0] * 0.55 + 50)),
            min(255, int(acc[1] * 0.55 + 50)),
            min(255, int(acc[2] * 0.55 + 50)),
        )
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
    txt = text_cache.render(f_small, label, txt_col)
    surf.blit(txt, (rect.x + (rect.width - txt.get_width()) // 2, rect.y + (rect.height - txt.get_height()) // 2))
