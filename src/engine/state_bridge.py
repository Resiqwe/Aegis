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
        now = time.perf_counter()
        self._prev_state = None
        self._curr_state = None
        self._prev_timestamp = now
        self._curr_timestamp = now
        self._sim_tick_delta = 1.0 / 60.0
        self._ema_pub_dt: float = 1.0 / 60.0
        # Atomický neměnný rámec (prev, curr, t_curr, ema_pub_dt) pro 100% lock-free čtení z GUI
        self._state_frame: tuple | None = None
        self._latest_snapshot: dict | None = None
        self._snapshot_seq: int = 0
        self._last_consumed_seq: int = 0
        self._latest_snapshot_cache = None
        self._selected_unit_id: int | None = None
        self._auto_select_best: bool = True
        self._banners: list[str] = []

    def post_banner(self, message: str):
        """Vloží notifikační zprávu pro HUD do thread-safe fronty."""
        with self._lock:
            self._banners.append(str(message))

    def pop_banners(self) -> list[str]:
        """Vyjme a vrátí všechny čekající notifikační zprávy pro HUD."""
        with self._lock:
            if not self._banners:
                return []
            banners = self._banners
            self._banners = []
            return banners

    def set_selected_unit(self, unit_id: int | None, auto_select_best: bool = False):
        """Nastaví požadovanou sledovanou jednotku z UI vlákna pod CPU zámkem."""
        with self._lock:
            self._selected_unit_id = None if unit_id is None or unit_id < 0 else int(unit_id)
            self._auto_select_best = bool(auto_select_best) or (self._selected_unit_id is None)

    def get_selected_unit_request(self) -> tuple[int | None, bool]:
        """Vrátí (selected_unit_id, auto_select_best) pro SimWorker vlákno."""
        with self._lock:
            return self._selected_unit_id, self._auto_select_best

    @staticmethod
    def _states_compatible(prev: dict | None, curr: dict | None) -> bool:
        """Ověří shodu rozměrů tenzorů i identifikátoru operace/epochy před interpolací."""
        if prev is None or curr is None:
            return False
        pos_prev = prev.get("pos")
        pos_curr = curr.get("pos")
        if pos_prev is None or pos_curr is None:
            return False
        if getattr(pos_prev, "shape", None) != getattr(pos_curr, "shape", None):
            return False
        for key in ("operation_id", "current_epoch", "scenario_idx"):
            if key in prev and key in curr and prev[key] != curr[key]:
                return False
        return True

    def publish_simulation_state(self, raw_state: dict, fixed_dt: float = 1.0 / 60.0):
        """Volá worker simulační vlákno po dokončení kroků (atomický swap reference + Adaptive EMA dt)."""
        now = time.perf_counter()
        self._snapshot_seq += 1
        raw_state["_seq"] = self._snapshot_seq

        curr_prev = self._curr_state
        if self._states_compatible(curr_prev, raw_state):
            new_prev = curr_prev
            real_dt = now - self._curr_timestamp
            if 0.001 <= real_dt <= 0.25:
                clamped_dt = max(1.0 / 120.0, min(0.25, real_dt))
                self._ema_pub_dt = 0.80 * self._ema_pub_dt + 0.20 * clamped_dt
            self._prev_timestamp = self._curr_timestamp
        else:
            new_prev = raw_state
            self._prev_timestamp = now

        self._prev_state = new_prev
        self._curr_state = raw_state
        self._curr_timestamp = now
        self._sim_tick_delta = self._ema_pub_dt

        # Jediná atomická publikace neměnného páru i nejnovějšího snímku (lock-free pro GUI čtenáře)
        self._state_frame = (new_prev, raw_state, now, self._ema_pub_dt)
        self._latest_snapshot = raw_state

    def publish_snapshot(self, snapshot: dict, fixed_dt: float = 1.0 / 60.0) -> None:
        """Alias pro atomickou publikaci kompletního snapshotu (lock-free double buffer)."""
        self.publish_simulation_state(snapshot, fixed_dt=fixed_dt)

    def is_snapshot_consumed(self) -> bool:
        """Vrátí True, pokud si GUI vlákno již vyzvedlo poslední publikovaný snapshot."""
        return self._last_consumed_seq >= self._snapshot_seq

    def get_latest_snapshot(self) -> dict | None:
        """Vrátí poslední kompletní snapshot zcela bez zamykání (lock-free)."""
        snap = self._latest_snapshot
        if snap is not None:
            self._last_consumed_seq = snap.get("_seq", self._last_consumed_seq)
        return snap

    def get_interpolated_state(self) -> dict | None:
        """
        Volá renderovací smyčka (60–180 FPS) zcela bez zámků (lock-free).
        Čte atomický neměnný rámec (prev, curr, t_curr, ema_pub_dt) a provádí čistou
        Adaptive Lerp interpolaci v rozsahu [0.0, 1.0] bez přestřelování.
        """
        frame = self._state_frame
        if frame is None:
            return None

        prev, curr, t_curr, ema_dt = frame
        if curr is None:
            return None

        self._last_consumed_seq = curr.get("_seq", self._last_consumed_seq)

        # Pokud ještě nemáme 2 po sobě jdoucí stavy, vrátíme aktuální
        if prev is None or prev is curr or ema_dt <= 0.0:
            return curr

        pos_prev = prev.get("pos")
        pos_curr = curr.get("pos")

        # Ochrana před změnou počtu jednotek (N) nebo přechodem na novou operaci/epochu
        if not self._states_compatible(prev, curr):
            snapshot = dict(curr)
            snapshot["pos"] = pos_curr
            snapshot["heading"] = curr.get("heading")
            snapshot["is_interpolated"] = False
            snapshot["lerp_alpha"] = 1.0
            return snapshot

        now = time.perf_counter()
        # Adaptive Lerp: čistý rozsah [0.0, 1.0] podle reálného klouzavého intervalu publikace
        time_since_curr = now - t_curr
        alpha = float(np.clip(time_since_curr / max(0.001, ema_dt), 0.0, 1.0))

        # Plynulý Lerp pozic: pos = (1 - a) * prev + a * curr
        interp_pos = (1.0 - alpha) * pos_prev + alpha * pos_curr

        # Interpolace úhlů (heading) přes nejkratší oblouk
        h_prev = prev.get("heading")
        h_curr = curr.get("heading")
        if (
            h_prev is not None
            and h_curr is not None
            and getattr(h_prev, "shape", None) == getattr(h_curr, "shape", None)
        ):
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
