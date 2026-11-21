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
from scenarios import MapScenario, compute_cpu_flow_field, SCENARIO_DOMINATION, SCENARIO_CITADEL, SCENARIO_NAMES
from engine.staging_buffer import SimulationStagingBuffer
from engine.tick_context import TickContext
from engine.sensors_system import run_sensors_and_brain
from engine.locomotion_system import run_locomotion
from engine.campaign import (
    start_new_operation as _campaign_start_new_operation,
    finish_operation_and_evolve as _campaign_finish_operation_and_evolve,
    compute_general_orders as _campaign_compute_general_orders,
)
from engine.spatial_system import SpatialSystem, check_line_of_sight
from engine.ballistics_system import VectorizedBallisticsSystem
from engine.combat_system import CombatSystem, run_weapons_and_firing
from engine.objectives_system import ObjectivesSystem, run_post_commit_and_objectives
from ai_brains import (
    NeuroEvolutionPool,
    CLASS_GRUNT, CLASS_MEDIC, CLASS_SNIPER, CLASS_JUGGERNAUT,
    CLASS_COSTS, UNLOCK_XP
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
        """
        Fáze 1 (Stage): Minimální kritická sekce pod _engine_lock (< 0.3 ms).
        Stáhne sdružený balík GPU tenzorů do CPU paměti a oddělí měnitelné Python kolekce.
        """
        with self._engine_lock:
            alive = self.hp > 0.0
            N = self.N

            # Sdružený balík 21 sloupců: 1 jediný PCIe přenos pro všechny per-unit atributy
            packed_gpu = torch.empty((N, 21), dtype=torch.float32, device=self.device)
            packed_gpu[:, 0] = self.pos[:, 0]
            packed_gpu[:, 1] = self.pos[:, 1]
            packed_gpu[:, 2] = self.heading
            packed_gpu[:, 3] = self.hp
            packed_gpu[:, 4] = self.class_max_hp
            packed_gpu[:, 5] = self.teams.float()
            packed_gpu[:, 6] = self.classes.float()
            packed_gpu[:, 7] = alive.float()
            packed_gpu[:, 8] = self.suppression
            packed_gpu[:, 9] = self.in_cover.float()
            packed_gpu[:, 10] = self.reload_timer
            packed_gpu[:, 11] = self.has_kinetic_shield.float() if hasattr(self, "has_kinetic_shield") else 0.0
            packed_gpu[:, 12] = self.unit_state.float() if hasattr(self, "unit_state") else 0.0
            packed_gpu[:, 13] = self.damage_dealt
            packed_gpu[:, 14] = self.damage_taken
            packed_gpu[:, 15] = self.unit_kills.float()
            packed_gpu[:, 16] = self.fitness
            packed_gpu[:, 17] = self.healing_done
            packed_gpu[:, 18] = self.ammo_in_mag
            packed_gpu[:, 19] = self.class_mag_size
            packed_gpu[:, 20] = self.class_range

            packed_cpu = packed_gpu.detach().cpu().numpy()

            # Efekty a mapové tenzory (jednorázové stažení na CPU)
            tracers_p1_np = self.active_tracers_p1.detach().cpu().numpy()
            tracers_p2_np = self.active_tracers_p2.detach().cpu().numpy()
            tracers_team_np = self.active_tracers_team.detach().cpu().numpy()
            heals_p1_np = self.active_heals_p1.detach().cpu().numpy()
            heals_p2_np = self.active_heals_p2.detach().cpu().numpy()
            sparks_np = self.hit_sparks.detach().cpu().numpy()
            armor_sparks_np = self.armor_sparks.detach().cpu().numpy()
            sniper_lasers_np = self.sniper_lasers.detach().cpu().numpy()
            smokes_np = self.active_smokes_gpu.detach().cpu().numpy()
            nodes_np = self.nodes_gpu.detach().cpu().numpy()
            node_pts_np = self.node_control_points.detach().cpu().numpy()
            kills_per_faction_np = self.kills_per_faction.detach().cpu().numpy()
            gate_hp_np = self.gate_hp.detach().cpu().numpy() if self.gate_hp.numel() > 0 else np.empty((0,), dtype=np.float32)

            # Kopie metadat generálů a bran (mimo mutaci při běhu další epochy)
            generals_raw = []
            for f in range(self.n_factions):
                gen = self.brains.generals[f]
                generals_raw.append({
                    "faction_id": f,
                    "level": int(gen.level),
                    "xp": float(gen.xp),
                    "next_xp": UNLOCK_XP[min(3, gen.level + 1)],
                    "unlocked": list(gen.get_unlocked_classes()),
                    "probs": list(gen.get_class_probabilities()),
                    "roi_ema": list(gen.class_roi_ema),
                })

            gates_raw = [
                {
                    "rect": list(g["rect"]),
                    "name": g.get("name", f"BRÁNA {gi + 1}"),
                    "max_hp": float(g.get("max_hp", 1500.0)),
                    "destroyed": bool(g.get("destroyed", False)),
                    "team": int(g.get("team", 0)),
                }
                for gi, g in enumerate(getattr(self.scenario, "gates", []))
            ]

            # Rychlá příprava inspekce sledované jednotky na CPU (bez 20× .item() synců)
            inspected_raw = None
            if include_inspection and N > 0:
                hp_np = packed_cpu[:, 3]
                fit_np = packed_cpu[:, 16]
                if auto_best or target_id is None or not (0 <= target_id < N) or hp_np[target_id] <= 0.0:
                    alive_mask = hp_np > 0.0
                    if np.any(alive_mask):
                        fit_masked = np.where(alive_mask, fit_np, -99999.0)
                        resolved_id = int(np.argmax(fit_masked))
                    else:
                        resolved_id = 0
                else:
                    resolved_id = int(target_id)

                if 0 <= resolved_id < N:
                    try:
                        b_slot = int(self.brain_slots[resolved_id].item())
                        flow = self.brains.inspect_decision_flow(b_slot, self.last_sensors[resolved_id])
                        tar_idx = int(self.last_target_enemy[resolved_id].item()) if hasattr(self, "last_target_enemy") else -1
                        has_los = bool(self.has_los_target[resolved_id].item()) if hasattr(self, "has_los_target") else False
                        inspected_raw = {
                            "unit_idx": resolved_id,
                            "brain_slot": b_slot,
                            "tar_idx": tar_idx,
                            "has_los": has_los,
                            "decision_flow": flow,
                        }
                    except Exception:
                        inspected_raw = None

            return {
                "N": int(N),
                "n_factions": int(self.n_factions),
                "packed_cpu": packed_cpu,
                "tracers_p1": tracers_p1_np,
                "tracers_p2": tracers_p2_np,
                "tracers_team": tracers_team_np,
                "heals_p1": heals_p1_np,
                "heals_p2": heals_p2_np,
                "sparks": sparks_np,
                "armor_sparks": armor_sparks_np,
                "sniper_lasers": sniper_lasers_np,
                "smokes": smokes_np,
                "nodes": nodes_np,
                "node_points": node_pts_np,
                "kills_per_faction": kills_per_faction_np,
                "gate_hp": gate_hp_np,
                "gates_raw": gates_raw,
                "generals_raw": generals_raw,
                "initial_team_counts": list(self.initial_team_counts),
                "particles": [dict(p) for p in self.particles],
                "walls": [list(w) for w in self.current_walls],
                "pillars": list(self.scenario.pillars),
                "bases": list(self.scenario.bases),
                "events": list(self.combat_events),
                "scenario_name": self.scenario.name,
                "scenario_idx": int(self.current_scenario_idx),
                "current_epoch": int(self.current_epoch),
                "operation_id": int(getattr(self, "operation_id", 0)),
                "operation_time_sec": float(self.operation_time_sec),
                "epoch_wins_per_faction": list(self.epoch_wins_per_faction),
                "faction_doctrines": list(self.faction_doctrines),
                "faction_techs": [list(t) for t in self.faction_active_techs],
                "history_avg_fitness": [list(h) for h in self.brains.history_avg_fitness],
                "history_best_fitness": [list(h) for h in self.brains.history_best_fitness],
                "inspected_raw": inspected_raw,
            }

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


def format_battle_snapshot(raw_state: dict) -> dict:
    """
    Fáze 2 (Format): Čistě CPU sestavení snapshotu pro HUD mimo zámek enginu.
    Nulový přístup k self.engine i GPU.
    """
    packed_cpu = raw_state["packed_cpu"]
    n_factions = raw_state["n_factions"]

    pos_np = packed_cpu[:, 0:2]
    heading_np = packed_cpu[:, 2]
    hp_np = packed_cpu[:, 3]
    max_hp_np = packed_cpu[:, 4]
    teams_np = packed_cpu[:, 5].astype(np.int8)
    classes_np = packed_cpu[:, 6].astype(np.int8)
    alive_np = packed_cpu[:, 7] > 0.5
    suppression_np = packed_cpu[:, 8]
    in_cover_np = packed_cpu[:, 9] > 0.5
    reloading_np = packed_cpu[:, 10] > 0.0
    kinetic_shield_np = packed_cpu[:, 11] > 0.5
    unit_state_np = packed_cpu[:, 12].astype(np.int32)
    dmg_dealt_np = packed_cpu[:, 13]
    dmg_taken_np = packed_cpu[:, 14]
    kills_np = packed_cpu[:, 15].astype(np.int32)
    fit_np = packed_cpu[:, 16]
    heal_np = packed_cpu[:, 17]

    generals_meta = []
    for g_raw in raw_state["generals_raw"]:
        f = g_raw["faction_id"]
        classes_in_team = classes_np[teams_np == f]
        comp = [int(np.sum(classes_in_team == c)) for c in range(4)]
        generals_meta.append({
            "faction_id": f,
            "level": g_raw["level"],
            "xp": g_raw["xp"],
            "next_xp": g_raw["next_xp"],
            "unlocked": g_raw["unlocked"],
            "composition": comp,
            "probs": g_raw["probs"],
            "roi_ema": g_raw["roi_ema"],
        })

    live_class_stats = []
    for c in range(4):
        c_mask = classes_np == c
        tot_cnt = int(np.sum(c_mask))
        if tot_cnt > 0:
            c_alive = int(np.sum(c_mask & alive_np))
            c_dmg = float(np.sum(dmg_dealt_np[c_mask]))
            c_taken = float(np.sum(dmg_taken_np[c_mask]))
            c_kills = int(np.sum(kills_np[c_mask]))
            c_heal = float(np.sum(heal_np[c_mask]))
            c_fit = float(np.mean(fit_np[c_mask]))
            c_roi = c_fit / float(CLASS_COSTS[c])
        else:
            c_alive, c_dmg, c_taken, c_kills, c_heal, c_fit, c_roi = 0, 0.0, 0.0, 0, 0.0, 0.0, 0.0

        live_class_stats.append({
            "class_id": c,
            "name": CLASS_NAMES[c],
            "cost": CLASS_COSTS[c],
            "total": tot_cnt,
            "alive": c_alive,
            "damage": c_dmg,
            "damage_taken": c_taken,
            "kills": c_kills,
            "healing": c_heal,
            "avg_fitness": c_fit,
            "roi": c_roi,
        })

    gate_hp_np = raw_state["gate_hp"]
    gates_meta = []
    for gi, g in enumerate(raw_state["gates_raw"]):
        gates_meta.append({
            "rect": g["rect"],
            "name": g["name"],
            "hp": float(gate_hp_np[gi]) if gi < len(gate_hp_np) else 0.0,
            "max_hp": g["max_hp"],
            "destroyed": g["destroyed"],
            "team": g["team"],
        })

    alive_counts = [int(np.sum(alive_np[teams_np == f])) for f in range(n_factions)]
    kills_by_faction = [int(k) for k in raw_state["kills_per_faction"].tolist()]
    initial_counts = raw_state["initial_team_counts"]
    deaths_by_faction = [
        max(0, int(initial_counts[f]) - alive_counts[f])
        for f in range(n_factions)
    ]
    alive_by_class = [
        [int(np.sum(alive_np[(teams_np == f) & (classes_np == c)])) for c in range(4)]
        for f in range(n_factions)
    ]

    return {
        "pos": pos_np,
        "heading": heading_np,
        "alive_counts": alive_counts,
        "alive_by_class": alive_by_class,
        "kills_by_faction": kills_by_faction,
        "deaths_by_faction": deaths_by_faction,
        "initial_team_counts": initial_counts,
        "hp": hp_np,
        "max_hp": max_hp_np,
        "teams": teams_np,
        "classes": classes_np,
        "alive": alive_np,
        "suppression": suppression_np,
        "in_cover": in_cover_np,
        "reloading": reloading_np,
        "tracers_p1": raw_state["tracers_p1"],
        "tracers_p2": raw_state["tracers_p2"],
        "tracers_team": raw_state["tracers_team"],
        "heals_p1": raw_state["heals_p1"],
        "heals_p2": raw_state["heals_p2"],
        "sparks": raw_state["sparks"],
        "armor_sparks": raw_state["armor_sparks"],
        "sniper_lasers": raw_state["sniper_lasers"],
        "smokes": raw_state["smokes"],
        "particles": raw_state["particles"],
        "nodes": raw_state["nodes"],
        "node_points": raw_state["node_points"],
        "walls": raw_state["walls"],
        "pillars": raw_state["pillars"],
        "bases": raw_state["bases"],
        "gates": gates_meta,
        "events": raw_state["events"],
        "generals": generals_meta,
        "live_class_stats": live_class_stats,
        "scenario_name": raw_state["scenario_name"],
        "scenario_idx": raw_state["scenario_idx"],
        "current_epoch": raw_state["current_epoch"],
        "operation_id": raw_state["operation_id"],
        "operation_time_sec": raw_state["operation_time_sec"],
        "N": raw_state["N"],
        "n_factions": n_factions,
        "epoch_wins_per_faction": raw_state["epoch_wins_per_faction"],
        "faction_doctrines": raw_state["faction_doctrines"],
        "faction_techs": raw_state["faction_techs"],
        "has_kinetic_shield": kinetic_shield_np,
        "unit_state": unit_state_np,
        "history_avg_fitness": raw_state["history_avg_fitness"],
        "history_best_fitness": raw_state["history_best_fitness"],
    }


def format_inspected_unit(raw_state: dict) -> dict | None:
    """
    Fáze 2 (Format): Čistě CPU formátování inspekce vybrané jednotky mimo zámek enginu.
    """
    insp = raw_state.get("inspected_raw")
    if insp is None:
        return None

    packed_cpu = raw_state["packed_cpu"]
    N = raw_state["N"]
    uid = insp["unit_idx"]
    if not (0 <= uid < N):
        return None

    c_id = int(packed_cpu[uid, 6])
    t_id = int(packed_cpu[uid, 5])
    u_state = int(packed_cpu[uid, 12])
    u_th = float(packed_cpu[uid, 2])

    tar_idx = insp["tar_idx"]
    has_los = insp["has_los"]
    tar_pos = None
    if 0 <= tar_idx < N and packed_cpu[tar_idx, 3] > 0.0 and has_los:
        p_self = packed_cpu[uid, 0:2]
        p_tar = packed_cpu[tar_idx, 0:2]
        dx = float(p_tar[0] - p_self[0])
        dy = float(p_tar[1] - p_self[1])
        t_dist = math.hypot(dx, dy)
        t_ang = math.atan2(dy, dx)
        a_diff = (t_ang - u_th + math.pi) % (2.0 * math.pi) - math.pi
        if (abs(a_diff) < math.radians(72) or t_dist < 75.0) and t_dist < 700.0:
            tar_pos = [float(p_tar[0]), float(p_tar[1])]

    return {
        "unit_idx": uid,
        "brain_slot": insp["brain_slot"],
        "team": t_id,
        "class_id": c_id,
        "class_name": CLASS_NAMES[c_id],
        "pos": packed_cpu[uid, 0:2].copy(),
        "heading": u_th,
        "fitness": float(packed_cpu[uid, 16]),
        "kills": int(packed_cpu[uid, 15]),
        "damage_dealt": float(packed_cpu[uid, 13]),
        "damage_taken": float(packed_cpu[uid, 14]),
        "healing_done": float(packed_cpu[uid, 17]),
        "range": float(packed_cpu[uid, 20]),
        "target_pos": tar_pos,
        "hp": float(packed_cpu[uid, 3]),
        "max_hp": float(packed_cpu[uid, 4]),
        "ammo": float(packed_cpu[uid, 18]),
        "max_ammo": float(packed_cpu[uid, 19]),
        "reloading": bool(packed_cpu[uid, 10] > 0.0),
        "suppression": float(packed_cpu[uid, 8]),
        "in_cover": bool(packed_cpu[uid, 9] > 0.5),
        "unit_state": u_state,
        "unit_state_name": STATE_NAMES[u_state] if 0 <= u_state < len(STATE_NAMES) else "IDLE",
        "decision_flow": insp["decision_flow"],
    }
