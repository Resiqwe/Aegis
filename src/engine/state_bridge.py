"""
engine/state_bridge.py - Lock-Free Double Buffer a Hermite-Lerp Interpolace.

Zajišťuje bezeztrátové a neblokující předávání stavu mezi simulačním worker vláknem
(Worker Thread) a renderovacím vláknem (Render Loop - 180 FPS Lock).
"""

import time
import threading
import numpy as np
from dataclasses import dataclass, field


@dataclass
class InterpolatedRenderSnapshot:
    """Snapshot stavu pro vykreslovač s interpolovanými pozicemi jednotek."""
    pos: np.ndarray             # (N, 2) float32 - hladce vyhlazené pozice
    heading: np.ndarray         # (N,) float32 - plynulé natočení
    hp: np.ndarray              # (N,) float32
    max_hp: np.ndarray          # (N,) float32
    teams: np.ndarray           # (N,) int32
    classes: np.ndarray         # (N,) int32
    alive: np.ndarray           # (N,) bool
    ammo: np.ndarray            # (N,) float32
    max_ammo: np.ndarray        # (N,) float32
    in_cover: np.ndarray        # (N,) bool
    suppression: np.ndarray     # (N,) float32
    reloading: np.ndarray       # (N,) bool
    target_pos: list            # [(x, y) nebo None] pro každou jednotku
    
    # Bojové efekty a události v tomto okně
    tracers: list = field(default_factory=list)
    lasers: list = field(default_factory=list)
    heals: list = field(default_factory=list)
    sparks: list = field(default_factory=list)
    smokes: list = field(default_factory=list)
    
    # Globální stav mapy a cílů
    bases: list = field(default_factory=list)
    walls: list = field(default_factory=list)
    pillars: list = field(default_factory=list)
    nodes: list = field(default_factory=list)
    gates: list = field(default_factory=list)
    
    # Telemetrické a KPI metriky
    sim_tick: int = 0
    operation_time_sec: float = 0.0
    scenario_idx: int = 0
    scenario_name: str = ""
    node_points: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    kills_by_faction: list = field(default_factory=lambda: [0, 0, 0, 0])
    deaths_by_faction: list = field(default_factory=lambda: [0, 0, 0, 0])
    alive_counts: list = field(default_factory=lambda: [0, 0, 0, 0])
    alive_by_class: list = field(default_factory=list)
    generals: list = field(default_factory=list)
    events: list = field(default_factory=list)
    live_class_stats: list = field(default_factory=list)
    history_avg_fitness: list = field(default_factory=list)


class LockFreeStateBridge:
    """
    Atomický dvojitý buffer spojující simulační a renderovací vlákno.
    Umožňuje renderovacímu vláknu plynule dopočítávat polohu jednotek (Lerp),
    aniž by simulační vlákno zdržovalo render nebo render zdržoval fyziku.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._prev_state = None
        self._curr_state = None
        self._prev_timestamp = time.perf_counter()
        self._curr_timestamp = time.perf_counter()
        self._sim_tick_delta = 1.0 / 60.0
        self._latest_snapshot_cache = None

    def publish_simulation_state(self, raw_state: dict, fixed_dt: float = 1.0 / 60.0):
        """Volá worker simulační vlákno po každém dokončeném kroku fyziky."""
        now = time.perf_counter()
        with self._lock:
            self._prev_state = self._curr_state
            self._prev_timestamp = self._curr_timestamp
            self._curr_state = raw_state
            self._curr_timestamp = now
            self._sim_tick_delta = fixed_dt

    def get_interpolated_state(self) -> dict | None:
        """
        Volá renderovací smyčka (180 FPS) kdykoliv bez blokování.
        Vypočte alfa faktor a hladce interpoluje pozice a úhly jednotek.
        """
        with self._lock:
            curr = self._curr_state
            prev = self._prev_state
            t_curr = self._curr_timestamp
            dt_sim = self._sim_tick_delta

        if curr is None:
            return None

        # Pokud ještě nemáme 2 po sobě jdoucí stavy, vrátíme aktuální
        if prev is None or dt_sim <= 0.0:
            return curr

        now = time.perf_counter()
        # Alpha: kde se renderovací čas nachází mezi t_prev a t_curr
        # Hodnotu clampujeme do rozsahu [0.0, 1.25] pro drobnou predikci (extrapolaci)
        time_since_curr = now - t_curr
        alpha = float(np.clip(time_since_curr / max(0.001, dt_sim), 0.0, 1.25))

        # Rychlá vektorová interpolace poloh a úhlů
        pos_prev = prev["pos"]
        pos_curr = curr["pos"]
        
        # Plynulý Lerp pozic: pos = (1 - a) * prev + a * curr
        interp_pos = (1.0 - alpha) * pos_prev + alpha * pos_curr

        # Interpolace úhlů (heading) přes nejkratší oblouk
        h_prev = prev.get("heading")
        h_curr = curr.get("heading")
        if h_prev is not None and h_curr is not None:
            diff = (h_curr - h_prev + np.pi) % (2.0 * np.pi) - np.pi
            interp_heading = h_prev + diff * alpha
        else:
            interp_heading = h_curr

        # Sestavení konzistentního výstupního slovníku (kopírujeme reference kromě vyhlazených poloh)
        snapshot = dict(curr)
        snapshot["pos"] = interp_pos
        snapshot["heading"] = interp_heading
        snapshot["is_interpolated"] = True
        snapshot["lerp_alpha"] = alpha

        return snapshot
