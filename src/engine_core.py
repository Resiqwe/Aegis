"""
engine_core.py - Aegis Decoupled Engine 2.0 s Hierarchickým Genofondem a Deterministickou Pipeline
Klíčové metodické a technické vlastnosti:
  1. Třídně Izolovaný Genofond (4 x 4 x 50 = 800 Slotů v GPU VRAM):
     - Pěšák, Medik, Sniper a Juggernaut mají izolované evoluční sub-banky (žádná mezidruhová kontaminace).
  2. Hierarchický 2-Vrstvý MLP (14 Senzorů -> 64 H1 Percepce -> 64 H2 Taktika -> 7 Holonomických Akcí):
     - Surge, Sway, Yaw, Trigger, Stance (Sprint/Plížení), Ability (Třídní schopnost), Squad Ping.
  3. Aegis 2.0 Vektorová Balistika & Kinetika:
     - Úhlová redukce pancíře (dot product), zónový rozpad (hlava 2.5x, trup 1.0x, paže 0.55x, nohy 0.65x).
     - Kinetický Ragdoll a třídní prahy: Juggernaut ignoruje stagger pod 95 J, Sniper má garantovaný knockback.
  4. Zero-Trust Staged Pipeline:
     - Žádné rozptýlené přepisy stavu jednotek. Subsystémy zapisují delty do SimulationStagingBufferu.
     - Atomický commit v jediném vektorovém kroku garantuje determinismus a eliminuje CPU-GPU jitter.
  5. Vektorový stavový automat rolí (Maticový ArgMax registr):
     - Sniper lock při zaměřování, Medic prioritní fronta léčení, Grunt útočný/krycí manévr, Juggernaut fortress push.
"""

import math
import threading
import numpy as np
import torch

from config import CLASS_NAMES
from scenarios import MapScenario, compute_cpu_flow_field, SCENARIO_DOMINATION
from engine.staging_buffer import SimulationStagingBuffer
from engine.tick_context import TickContext
from engine.sensors_system import run_sensors_and_brain
from engine.locomotion_system import run_locomotion
from engine.campaign import (
    start_new_operation as _campaign_start_new_operation,
    finish_operation_and_evolve as _campaign_finish_operation_and_evolve,
    compute_general_orders as _campaign_compute_general_orders,
)
from engine.snapshot_format import (
    extract_raw_telemetry_state as _extract_raw_telemetry_state,
    format_battle_snapshot,
    format_inspected_unit,
)
from engine.spatial_system import SpatialSystem
from engine.ballistics_system import VectorizedBallisticsSystem
from engine.combat_system import CombatSystem, run_weapons_and_firing
from engine.objectives_system import ObjectivesSystem, run_post_commit_and_objectives
from ai_brains import (
    NeuroEvolutionPool
)

FACTION_COLORS = [
    [0.0, 0.94, 0.82],   # ALPHA: Tyrkysová
    [1.0, 0.18, 0.45],   # OMEGA: Růžovo-červená
    [1.0, 0.76, 0.10],   # SOLAR: Zlato-žlutá
    [0.69, 0.31, 0.98],  # PHANTOM: Fialová
]

DOCTRINE_BALANCED = 0
DOCTRINE_RUSH = 1
DOCTRINE_HOLD = 2
DOCTRINE_SKIRMISH = 3

DOCTRINE_LIST = ["BALANCED", "RUSH", "HOLD", "SKIRMISH"]

# Konstanty pro Maticový stavový registr (ArgMax State Registry):
STATE_IDLE = 0            # Základní pohyb dle mozku a doktríny
STATE_ENGAGE = 1          # Frontální útok a střelba (Grunt nápor, palebná převaha)
STATE_COVER_RETREAT = 2   # Ústup do krytu (reakce na těžký kinetický impuls / poškození)
STATE_SNIPER_LOCK = 3     # Zastavení pohybu, uzamčení zaměřovače a výstřel z místa
STATE_TRIAGE_HEAL = 4     # Medik prioritní fronta léčení bez útočných pokusů
STATE_FORTRESS_PUSH = 5   # Juggernaut těžký postup vpřed ignorující lehký stagger
STATE_NAMES = ["IDLE", "ENGAGE", "COVER_RETREAT", "SNIPER_LOCK", "TRIAGE_HEAL", "FORTRESS_PUSH"]


class RealisticBattleEngine:
    MAX_OPERATION_DURATION = 75.0

    def __init__(self, width: int = 2400, height: int = 1350, device: torch.device | None = None):
        # Thread-safe reentrantní zámek chránící stav tenzorů před souběžným zápisem
        self._engine_lock = threading.RLock()
        with self._engine_lock:
            self.W = width
            self.H = height
            self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            self.N = 400
            self.n_factions = 4
            self.budget_per_faction = 100

            self.CELL_SIZE = 32
            self.GW = (self.W + self.CELL_SIZE - 1) // self.CELL_SIZE
            self.GH = (self.H + self.CELL_SIZE - 1) // self.CELL_SIZE

            self.brains = NeuroEvolutionPool(self.device)
            self.ballistics = VectorizedBallisticsSystem(self.device)
            self.spatial = SpatialSystem(self.W, self.H, self.device)
            self.combat = CombatSystem(self.device)
            self.objectives = ObjectivesSystem(self.device)

            self.current_scenario_idx = SCENARIO_DOMINATION
            self.scenario = MapScenario(self.current_scenario_idx, self.W, self.H)
            self.current_walls = [list(w) for w in self.scenario.walls]
            self.current_pillars = [list(p) for p in self.scenario.pillars]
            self.gate_wall_indices = list(getattr(self.scenario, "gate_wall_indices", []))
            self._reset_gates()
            self.walls_gpu, self.pillars_gpu, self.nodes_gpu = self.scenario.build_gpu_tensors(self.device)
            self.faction_colors_gpu = torch.tensor(FACTION_COLORS, dtype=torch.float32, device=self.device)

            self._rebuild_flow_field()

            self.telemetry_mgr = None
            self.operation_time_sec = 0.0
            self.tick_counter = 0
            self.current_epoch = 1
            self.operation_id = 0
            self.epoch_winner = -1
            self.epoch_end_reason = ""

            self.faction_doctrines = [0, 0, 0, 0]
            self.faction_doctrine_tensor = torch.zeros((4,), dtype=torch.long, device=self.device)
            self.kills_per_faction = torch.zeros((4,), dtype=torch.int32, device=self.device)
            self.node_control_points = torch.zeros((4,), dtype=torch.float32, device=self.device)
            self.epoch_wins_per_faction = [0, 0, 0, 0]

            # Vizuální a částicové efekty
            self.active_tracers_p1 = torch.zeros((0, 2), dtype=torch.float32, device=self.device)
            self.active_tracers_p2 = torch.zeros((0, 2), dtype=torch.float32, device=self.device)
            self.active_tracers_team = torch.zeros((0,), dtype=torch.int8, device=self.device)
            self.active_heals_p1 = torch.zeros((0, 2), dtype=torch.float32, device=self.device)
            self.active_heals_p2 = torch.zeros((0, 2), dtype=torch.float32, device=self.device)
            self.hit_sparks = torch.zeros((0, 2), dtype=torch.float32, device=self.device)
            self.armor_sparks = torch.zeros((0, 2), dtype=torch.float32, device=self.device)
            self.sniper_lasers = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
            self.particles = []  # Částice výbuchů a trosek
            self.combat_events = []
            # Vizuální efekty lze vypnout v headless/background režimu pro maximální propustnost.
            self.visual_mode = True
            # Zero-allocation pracovní buffer pro senzory (realokuje se jen při změně N).
            self._sensor_buf = None
            # Zero-Trust Staging Buffer pro oddělení výpočtů od zápisů
            self.staging_buf = SimulationStagingBuffer(self.N, self.device)
            
            # Prealokované konstantní tenzory pro eliminaci dynamické CUDA alokace v hot-path
            self.CONST_CLASS_MAX_VIS = torch.tensor([380.0, 320.0, 680.0, 380.0], dtype=torch.float32, device=self.device)
            self.CONST_RELOAD_TIME = torch.tensor(1.8, dtype=torch.float32, device=self.device)
            self.CONST_CLASS_COOLDOWNS = torch.tensor([8, 9, 22, 5], dtype=torch.int32, device=self.device)
            self.CONST_CLASS_BASE_ACC = torch.tensor([0.82, 0.80, 0.90, 0.78], dtype=torch.float32, device=self.device)

    def _reset_gates(self):
        gates = getattr(self.scenario, "gates", [])
        for g in gates:
            g["destroyed"] = False
            g["hp"] = float(g.get("max_hp", g.get("hp", 1500.0)))
        if gates:
            self.gate_hp = torch.tensor([g["hp"] for g in gates], dtype=torch.float32, device=self.device)
        else:
            self.gate_hp = torch.zeros((0,), dtype=torch.float32, device=self.device)

    def _rebuild_walls_gpu(self):
        if self.current_walls:
            self.walls_gpu = torch.tensor(self.current_walls, dtype=torch.float32, device=self.device)
        else:
            self.walls_gpu = torch.zeros((0, 4), dtype=torch.float32, device=self.device)

    def _rebuild_flow_field(self):
        walls = self.current_walls if hasattr(self, "current_walls") else self.scenario.walls
        pillars = self.current_pillars if hasattr(self, "current_pillars") else self.scenario.pillars
        field_np = compute_cpu_flow_field(self.W, self.H, walls, pillars, self.CELL_SIZE)
        self.flow_field_gpu = torch.tensor(field_np, dtype=torch.float32, device=self.device)

    def switch_scenario(self, scenario_idx: int):
        with self._engine_lock:
            self.current_scenario_idx = scenario_idx % 3
            self.scenario = MapScenario(self.current_scenario_idx, self.W, self.H)
            self.current_walls = [list(w) for w in self.scenario.walls]
            self.current_pillars = [list(p) for p in self.scenario.pillars]
            self.gate_wall_indices = list(getattr(self.scenario, "gate_wall_indices", []))
            self._reset_gates()
            self.walls_gpu, self.pillars_gpu, self.nodes_gpu = self.scenario.build_gpu_tensors(self.device)
            self._rebuild_flow_field()
            self.start_new_operation(restart_all=False)

    def configure(self, n_units: int = 400, n_factions: int = 4):
        with self._engine_lock:
            self.n_factions = max(2, min(4, int(n_factions)))
            self.budget_per_faction = max(50, min(self.brains.SLOTS_PER_FACTION, int(n_units) // self.n_factions))
            self.start_new_operation(restart_all=True)

    def start_new_operation(self, restart_all: bool = False):
        _campaign_start_new_operation(self, restart_all=restart_all)

    def set_faction_doctrine(self, faction_id: int, doctrine_idx: int):
        with self._engine_lock:
            if 0 <= faction_id < 4:
                self.faction_doctrines[faction_id] = int(doctrine_idx) % 4
                self.faction_doctrine_tensor[faction_id] = self.faction_doctrines[faction_id]

    def _check_battle_status(self, alive: torch.Tensor) -> bool:
        with self._engine_lock:
            finished, winner, reason = self.objectives.check_battle_status(
                self.operation_time_sec,
                self.teams,
                alive,
                self.initial_team_counts,
                self.kills_per_faction,
                self.node_control_points,
                self.n_factions,
                self.MAX_OPERATION_DURATION,
            )
            if finished:
                self.epoch_winner = winner
                self.epoch_end_reason = reason
                return True
            return False

    def finish_operation_and_evolve(self):
        _campaign_finish_operation_and_evolve(self)

    def _compute_general_orders(self, dir_enemy: torch.Tensor, dist_enemy: torch.Tensor) -> torch.Tensor:
        return _campaign_compute_general_orders(self, dir_enemy, dist_enemy)

    def step(self, dt: float = 0.0166):
        with self._engine_lock:
            self._step_internal(dt)

    def _step_internal(self, dt: float = 0.0166):
        self.operation_time_sec += dt
        self.tick_counter += 1
        N = self.N
        alive = self.hp > 0.0

        if (self.tick_counter % 10 == 0 or self.operation_time_sec >= self.MAX_OPERATION_DURATION) and self._check_battle_status(alive):
            self.finish_operation_and_evolve()
            return

        alive_idx = torch.where(alive)[0]
        K = alive_idx.numel()
        ctx = TickContext(dt=dt, N=N, alive=alive, alive_idx=alive_idx, K=K)

        # Reset Zero-Trust Staging Bufferu na začátku ticku
        self.staging_buf.reset()

        # Timery
        reloading = self.reload_timer > 0.0
        self.reload_timer = torch.where(reloading, self.reload_timer - dt, torch.zeros_like(self.reload_timer))
        just_reloaded = reloading & (self.reload_timer <= 0.0)
        self.ammo_in_mag = torch.where(just_reloaded, self.class_mag_size, self.ammo_in_mag)
        self.suppression = (self.suppression - dt * 0.35).clamp(min=0.0)
        # Power-Up: TECH_NANITES (0) - Pasivní nanitová regenerace (+3.5 HP/s mimo potlačení)
        for f in range(self.n_factions):
            if 0 in self.faction_active_techs[f]:
                nanite_mask = alive & (self.teams == f) & (self.suppression < 0.20)
                if nanite_mask.any():
                    self.hp[nanite_mask] = torch.minimum(self.class_max_hp[nanite_mask], self.hp[nanite_mask] + 3.5 * dt)

        # Kontrola krytu u zdi
        g_x = (self.pos[:, 0].long() // self.CELL_SIZE).clamp(0, self.GW - 1)
        g_y = (self.pos[:, 1].long() // self.CELL_SIZE).clamp(0, self.GH - 1)
        wall_force = self.flow_field_gpu[g_y, g_x]
        ctx.wall_force = wall_force
        self.in_cover = torch.norm(wall_force, dim=1) > 0.38

        # Taktická odměna za setrvání v krytu pod palbou
        under_fire_in_cover = alive & self.in_cover & (self.suppression > 0.25)
        self.fitness = torch.where(under_fire_in_cover, self.fitness + 0.15, self.fitness)

        # Fáze 1: Detekce nepřátel/spojenců, léčení Medika, egocentrické senzory a forward mozku
        run_sensors_and_brain(self, ctx)

        # Fáze 2: Pohyb, rotace, třídní schopnosti, stavový registr a FOV ztráta cíle
        run_locomotion(self, ctx)

        # Fáze 2.5: Přebíjení, střelba, vektorová balistika a zápis delt do staging bufferu
        run_weapons_and_firing(self, ctx)

        # ==============================================================
        # FÁZE 3: ATOMICKÝ HROMADNÝ COMMIT (Zero-Trust Pipeline)
        # Všechny akumulované delty ze staging bufferu aplikovány v jediném průchodu!
        # ==============================================================
        self._atomic_commit_pipeline(dt)

        # Fáze 4: Post-commit efekty (sniper lasery, anti-kemping, částice) a obsazování uzlů
        run_post_commit_and_objectives(self, ctx)

        self._update_gates(dt)

    def _update_gates(self, dt: float):
        """Obléhání zničitelných bran: delegováno do ObjectivesSystem (volající drží _engine_lock)."""
        gates = getattr(self.scenario, "gates", [])
        destroyed = self.objectives.update_gates(
            dt,
            gates,
            self.gate_hp,
            self.pos,
            self.teams,
            self.classes,
            self.hp > 0.0,
        )
        for gi in destroyed:
            self._destroy_gate(gi)

    def _destroy_gate(self, gate_idx: int):
        gates = getattr(self.scenario, "gates", [])
        if gate_idx >= len(gates):
            return
        gate = gates[gate_idx]
        gate["destroyed"] = True
        gate["hp"] = 0.0
        if self.visual_mode:
            gx = (gate["rect"][0] + gate["rect"][2]) * 0.5
            gy = (gate["rect"][1] + gate["rect"][3]) * 0.5
            for _ in range(24):
                self.particles.append({
                    "x": gx + float(np.random.randn() * 18.0),
                    "y": gy + float(np.random.randn() * 18.0),
                    "vx": float(np.random.randn() * 1.5),
                    "vy": float(np.random.randn() * 1.5),
                    "life": float(np.random.rand() * 2.0 + 2.0),
                    "col": (150, 140, 120),
                })
        wall_idx = self.gate_wall_indices[gate_idx] if gate_idx < len(self.gate_wall_indices) else -1
        if 0 <= wall_idx < len(self.current_walls):
            self.current_walls.pop(wall_idx)
            self.gate_wall_indices = [
                (idx - 1 if idx > wall_idx else idx) for idx in self.gate_wall_indices
            ]
            self._rebuild_walls_gpu()
            self._rebuild_flow_field()
        if len(self.combat_events) > 35:
            self.combat_events.pop(0)
        sec = int(self.operation_time_sec)
        self.combat_events.append({
            "time": self.operation_time_sec,
            "text": f"[{sec:02d}s] {gate.get('name', 'BRÁNA')} ZNIČENA! Průchod otevřen.",
            "team": -1,
        })

    def _atomic_commit_pipeline(self, dt: float):
        """
        Fáze 3: Atomický commit ze SimulationStagingBufferu (volající drží _engine_lock).
        Jednorázová vektorová aplikace změn HP, rychlostí, potlačení a GPU akumulace eliminací.
        """
        buf = self.staging_buf

        # 1. Aplikace poškození/léčení do HP a detekce zabitých
        prev_hp = self.hp.clone()
        new_hp = self.hp + buf.delta_hp
        self.hp = torch.minimum(torch.maximum(new_hp, torch.zeros_like(new_hp)), self.class_max_hp)
        self.damage_taken += buf.damage_taken
        self.fitness -= buf.out_of_cover_penalty

        just_killed = (prev_hp > 0.0) & (self.hp <= 0.0)

        # 2. Aplikace potlačení s horním stropem 1.0
        self.suppression = torch.clamp(self.suppression + buf.delta_suppression, 0.0, 1.0)

        # 3. Motoric Stagger & Kinetický impuls (Knockback)
        new_stagger = torch.maximum(self.stagger_timer, buf.stagger_duration)
        self.stagger_timer = new_stagger
        stagger_damp = torch.where(self.stagger_timer > 0.0, 0.20, 1.0).unsqueeze(-1)
        self.vel = (self.vel + buf.delta_vel) * stagger_damp

        # 4. Hromadné zpracování eliminací na GPU + dávkový přenos pro telemetrii a HUD
        if just_killed.any():
            k_t = torch.where(just_killed)[0]
            self.fitness[k_t] -= 25.0

            shooters = buf.last_attacker_id[k_t].long()
            valid_mask = (shooters >= 0) & (shooters < self.N)
            valid_shooters = shooters[valid_mask]

            if valid_shooters.numel() > 0:
                ones = torch.ones_like(valid_shooters, dtype=self.unit_kills.dtype)
                self.unit_kills.index_add_(0, valid_shooters, ones)
                self.fitness.index_add_(
                    0,
                    valid_shooters,
                    torch.full_like(valid_shooters, 30.0, dtype=self.fitness.dtype),
                )
                killer_teams = self.teams[valid_shooters].long()
                faction_kills = torch.bincount(killer_teams, minlength=4).to(self.kills_per_faction.dtype)
                self.kills_per_faction[:4] += faction_kills[:4]

            # Dávkové stažení základních metadat obětí (bez per-kill .item() synchronizací)
            v_ids_cpu = k_t.detach().cpu().numpy()
            teams_k = self.teams[k_t].detach().cpu().numpy()
            classes_k = self.classes[k_t].detach().cpu().numpy()

            if self.telemetry_mgr is not None or self.visual_mode:
                pos_k = self.pos[k_t].detach().cpu().numpy()

                if self.telemetry_mgr is not None:
                    shooters_cpu = shooters.detach().cpu().numpy()
                    all_teams_cpu = self.teams.detach().cpu().numpy()
                    all_classes_cpu = self.classes.detach().cpu().numpy()
                    for ki in range(len(v_ids_cpu)):
                        s_id = int(shooters_cpu[ki])
                        if 0 <= s_id < self.N:
                            self.telemetry_mgr.record_kill_event(
                                epoch=self.current_epoch,
                                time_sec=self.operation_time_sec,
                                killer_team=int(all_teams_cpu[s_id]),
                                victim_team=int(teams_k[ki]),
                                killer_class=int(all_classes_cpu[s_id]),
                                victim_class=int(classes_k[ki]),
                                x=float(pos_k[ki, 0]),
                                y=float(pos_k[ki, 1]),
                            )

                if self.visual_mode:
                    for ki in range(len(v_ids_cpu)):
                        col_t = FACTION_COLORS[int(teams_k[ki])]
                        for _ in range(8):
                            ang_p = float(np.random.rand() * 2 * math.pi)
                            spd_p = float(np.random.rand() * 3.5 + 1.0)
                            self.particles.append({
                                "x": float(pos_k[ki, 0]),
                                "y": float(pos_k[ki, 1]),
                                "vx": math.cos(ang_p) * spd_p,
                                "vy": math.sin(ang_p) * spd_p,
                                "life": 1.0,
                                "col": (int(col_t[0] * 255), int(col_t[1] * 255), int(col_t[2] * 255)),
                            })

            if len(self.combat_events) > 35:
                self.combat_events.pop(0)
            t_v = int(teams_k[0])
            c_v = CLASS_NAMES[int(classes_k[0])]
            meta_v = ("ALPHA", "OMEGA", "SOLAR", "PHANTOM")[t_v]
            sec = int(self.operation_time_sec)
            self.combat_events.append({
                "time": self.operation_time_sec,
                "text": f"[{sec:02d}s] {meta_v} ({c_v}) #{int(v_ids_cpu[0]):03d} vyřazen z boje",
                "team": t_v,
            })

    def extract_raw_telemetry_state(
        self,
        target_id: int | None = None,
        auto_best: bool = True,
        include_inspection: bool = True,
    ) -> dict:
        return _extract_raw_telemetry_state(
            self,
            target_id=target_id,
            auto_best=auto_best,
            include_inspection=include_inspection,
        )

    def get_battle_state_snapshot(self) -> dict:
        """Synchronní obálka pro zpětnou kompatibilitu (testy a single-thread režim)."""
        raw_state = self.extract_raw_telemetry_state(include_inspection=False)
        return format_battle_snapshot(raw_state)

    def get_best_alive_unit(self) -> int:
        with self._engine_lock:
            alive = self.hp > 0.0
            if not alive.any():
                return 0
            fit_masked = torch.where(alive, self.fitness, torch.full_like(self.fitness, -99999.0))
            return int(torch.argmax(fit_masked).item())

    def inspect_unit(self, unit_idx: int) -> dict | None:
        """Synchronní inspekce jednotky přes sdružený Stage -> Format pipeline."""
        raw_state = self.extract_raw_telemetry_state(
            target_id=unit_idx,
            auto_best=False,
            include_inspection=True,
        )
        return format_inspected_unit(raw_state)
