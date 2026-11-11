"""
engine/async_runner.py - Samostatné asynchronní Worker Vlákno simulátoru.

Řídí přesný simulační takt (TPS: 30, 60, 120, 240) se striktně fixním fyzikálním dt = 1/60s.
Komunikuje s renderovacím vláknem pomocí LockFreeStateBridge bez zpomalování FPS.
"""

import time
import threading
from typing import Callable, Any
from engine.state_bridge import LockFreeStateBridge


class AsyncSimulationWorker:
    """
    Spouští a řídí běh fyziky a AI v dedikovaném systémovém vlákně.
    """
    def __init__(self, engine_instance: Any, bridge: LockFreeStateBridge):
        self.engine = engine_instance
        self.bridge = bridge

        self._thread = None
        self._running = False
        self._paused = False
        self._stop_requested = threading.Event()
        
        # Škálování rychlosti simulace (0.5x, 1x, 2x, 4x)
        self.speed_multiplier = 1.0
        self.fixed_dt = 1.0 / 60.0  # Fyzika má VŽDY 1/60s krok pro absolutní stabilitu
        
        # Telemetrie vlákna
        self.current_tps = 0.0
        self.step_time_ms = 0.0
        self._step_counter = 0
        self._tps_timer = time.perf_counter()

    def set_speed(self, multiplier: float):
        """Nastaví rychlost simulace (např. 0.5, 1.0, 2.0, 4.0)."""
        self.speed_multiplier = max(0.1, float(multiplier))

    def set_paused(self, paused: bool):
        """Pozastaví nebo obnoví výpočet fyziky."""
        self._paused = paused

    def is_running(self) -> bool:
        return self._running and not self._paused

    def start(self):
        """Spustí simulační vlákno."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._running = True
        self._paused = False
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="Aegis-SimWorker")
        self._thread.start()

    def stop(self, join_timeout: float = 1.0):
        """Ukončí simulační vlákno."""
        self._stop_requested.set()
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=join_timeout)
        self._thread = None

    def _worker_loop(self):
        """
        Vnitřní smyčka worker vlákna s precizním time-accumulator řízením.
        """
        last_time = time.perf_counter()
        accumulator = 0.0
        sim_dt = self.fixed_dt

        # Počáteční publikace stavu
        initial_state = self.engine.get_battle_state_snapshot()
        self.bridge.publish_simulation_state(initial_state, sim_dt)

        while not self._stop_requested.is_set():
            now = time.perf_counter()
            frame_dt = min(0.10, now - last_time)
            last_time = now

            if not self._paused:
                # Akumulujeme čas podle rychlostního násobiče (0.5x, 1x, 2x, 4x)
                accumulator += frame_dt * self.speed_multiplier

                # Vykonáme odpovídající počet diskrétních kroků fyziky se stabilním dt = 1/60s
                steps_done = 0
                max_substeps = max(1, int(round(self.speed_multiplier * 4)))

                while accumulator >= sim_dt and steps_done < max_substeps:
                    t_step0 = time.perf_counter()
                    
                    # 1. Krok jádra
                    self.engine.step(sim_dt)
                    
                    t_step1 = time.perf_counter()
                    self.step_time_ms = 0.85 * self.step_time_ms + 0.15 * ((t_step1 - t_step0) * 1000.0)

                    # 2. Publikace do Lock-free mostu
                    raw_snapshot = self.engine.get_battle_state_snapshot()
                    self.bridge.publish_simulation_state(raw_snapshot, sim_dt)

                    accumulator -= sim_dt
                    steps_done += 1
                    self._step_counter += 1

                # Ochrana proti zpoždění při minimalizaci okna / alt-tabu
                if accumulator > 0.20:
                    accumulator = 0.0

            # Měření aktuální frekvence TPS (ticks per second)
            now_tps = time.perf_counter()
            if now_tps - self._tps_timer >= 0.5:
                dur = max(0.001, now_tps - self._tps_timer)
                self.current_tps = self._step_counter / dur
                self._step_counter = 0
                self._tps_timer = now_tps

            # Krátké kooperativní uvolnění vlákna CPU (0.5 ms sleep)
            time.sleep(0.0005)
