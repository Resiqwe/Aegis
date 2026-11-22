"""
config.py - Centrální konfigurační konstanty simulace Cyber-Tactics 2.0.
Sjednocuje rozměry bojiště, parametry tříd, fyzikální a herní konstanty.
"""

# ── Rozměry herního světa ───────────────────────────────────────────
WORLD_W: int = 2400
WORLD_H: int = 1350
CELL_SIZE: int = 32

# ── Třídy jednotek ──────────────────────────────────────────────────
CLASS_GRUNT: int = 0
CLASS_MEDIC: int = 1
CLASS_SNIPER: int = 2
CLASS_JUGGERNAUT: int = 3

CLASS_NAMES = (
    "Pěšák (Grunt)",
    "Medik (Medic)",
    "Sniper (Odstřelovač)",
    "Juggernaut (Těžkooděnec)",
)
CLASS_COSTS = (1, 2, 3, 4)

# Základní atributy tříd [Grunt, Medic, Sniper, Juggernaut]
CLASS_MAX_HP = (115.0, 105.0, 70.0, 480.0)
CLASS_SPEED = (3.20, 3.40, 2.20, 2.50)
CLASS_RANGE = (240.0, 200.0, 340.0, 220.0)
CLASS_DMG = (28.0, 18.0, 48.0, 36.0)
CLASS_MAG_SIZE = (32.0, 24.0, 6.0, 90.0)
CLASS_COOLDOWN = (8, 9, 22, 5)  # ticky mezi výstřely
CLASS_MAX_VISIBILITY = (380.0, 320.0, 680.0, 380.0)

# Barvy tříd (pro vykreslování v UI a HUDu)
CLASS_COLORS = (
    (74, 144, 226),   # Grunt: Modrá
    (65, 235, 100),   # Medic: Zelená
    (255, 195, 25),   # Sniper: Žlutá/Zlatá
    (231, 76, 60),    # Juggernaut: Červená
)

# ── Frakce a doktríny ────────────────────────────────────────────────
FACTION_NAMES = (
    "ALPHA VANGUARD",
    "OMEGA SYNDICATE",
    "SOLAR CORE",
    "PHANTOM UNIT",
)
FACTION_SHORTS = ("ALPHA", "OMEGA", "SOLAR", "PHANTOM")

FACTION_COLORS_RGB = (
    (0, 240, 210),    # Alpha: Tyrkysová
    (255, 45, 115),   # Omega: Růžovo-červená
    (255, 195, 25),   # Solar: Zlato-žlutá
    (175, 80, 250),   # Phantom: Fialová
)

FACTION_COLORS_NORMALIZED = [
    [0.0, 0.94, 0.82],
    [1.0, 0.18, 0.45],
    [1.0, 0.76, 0.10],
    [0.69, 0.31, 0.98],
]

DOCTRINE_BALANCED: int = 0
DOCTRINE_RUSH: int = 1
DOCTRINE_HOLD: int = 2
DOCTRINE_SKIRMISH: int = 3
DOCTRINE_LIST = ("BALANCED", "RUSH", "HOLD", "SKIRMISH")

# ── Stavový automat jednotek ──────────────────────────────────────────
STATE_IDLE: int = 0
STATE_ENGAGE: int = 1
STATE_COVER_RETREAT: int = 2
STATE_SNIPER_LOCK: int = 3
STATE_TRIAGE_HEAL: int = 4
STATE_FORTRESS_PUSH: int = 5
STATE_NAMES = ("IDLE", "ENGAGE", "COVER_RETREAT", "SNIPER_LOCK", "TRIAGE_HEAL", "FORTRESS_PUSH")

# ── Technologie (Power-Upy generálů) ─────────────────────────────────
TECH_NANITES: int = 0     # Pasivní samoléčení (+3.5 HP/s)
TECH_RADAR: int = 1       # +25 % dosah senzorů a LoS
TECH_SHIELDS: int = 2     # 100% absorpce 1. smrtícího zásahu
TECH_OVERCLOCK: int = 3   # +25 % kapacita zásobníku
TECH_AEROSOL: int = 4     # +40 % poloměr kouře a maskování

TECH_NAMES = (
    "Nanitová Regenerace (+3.5 HP/s pasivní samoléčení)",
    "Bojový UAV Radar (+25 % dosah senzorů a LoS)",
    "Kinetické Štíty (100% absorpce 1. smrtícího zásahu)",
    "Magazínový Overclock (+25 % kapacita zásobníku)",
    "Termální Aerosol (+40 % poloměr kouře a maskování)",
)
TECH_COSTS = (25, 30, 35, 20, 20)

# ── Generálské hodnosti a XP ─────────────────────────────────────────
UNLOCK_XP = (0, 500, 1500, 3500, 8000, 18000, 35000, 70000, 120000, 200000, 350000)
COMMAND_RANKS = (
    "Kadet (Základní výcvik)",
    "Poručík (Polní podpora)",
    "Kapitán (Taktický přehled)",
    "Major (Koordinovaný úder)",
    "Plukovník (Strategická dominance)",
    "Generál (Operační mistrovství)",
    "Polní Maršál (Absolutní kontrola)",
    "Vojvoda (Vrchní velení)",
    "Grand Strateg (Válečný génius)",
    "Primarch (Neporazitelný hegemon)",
    "Apex Imperator (Kybernetická dominance)",
)

# ── Senzory a akce sítě (14 -> 24 -> 16 -> 7) ────────────────────────
SENSOR_NAMES = (
    "Cíl Vpřed/Vzad",
    "Cíl Vpravo/Vlevo",
    "Blízkost cíle",
    "Příchozí palba Vpřed/Vzad",
    "Příchozí palba Vpravo/Vlevo",
    "Taktický povel Vpřed/Vzad",
    "Taktický povel Vpravo/Vlevo",
    "Míra potlačení",
    "Zásobník munice",
    "Zdraví (HP)",
    "Kryt Vpřed/Vzad",
    "Kryt Vpravo/Vlevo",
    "Koheze a Ping čety",
    "Stav zóny dominance",
)
SENSOR_COUNT: int = len(SENSOR_NAMES)  # 14

ACTION_NAMES = (
    "Surge (Plyn/Couvání)",
    "Sway (Úkrok Vpravo/Vlevo)",
    "Yaw (Otáčení těla/Záměr)",
    "Trigger (Spoušť palby)",
    "Stance (Sprint vs Zakleknutí)",
    "Ability (Taktická dovednost)",
    "Squad Ping (Signál četě)",
)
ACTION_COUNT: int = len(ACTION_NAMES)  # 7

# ── Simulační parametry ──────────────────────────────────────────────
MAX_OPERATION_DURATION: float = 75.0
FIXED_DT: float = 1.0 / 60.0
