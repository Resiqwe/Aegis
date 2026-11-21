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
from engine.spatial_system import SpatialSystem, check_line_of_sight
from engine.ballistics_system import VectorizedBallisticsSystem
from engine.combat_system import CombatSystem
from engine.objectives_system import ObjectivesSystem
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
        with self._engine_lock:
            self.operation_id = getattr(self, "operation_id", 0) + 1
            self.operation_time_sec = 0.0
            self.tick_counter = 0
            self.epoch_winner = -1
            self.epoch_end_reason = ""
            self.kills_per_faction.zero_()
            self.node_control_points.zero_()
            self.particles.clear()

            # Obnova zničitelných bran a dynamických stěn pro novou bitvu
            if hasattr(self, "scenario"):
                self.current_walls = [list(w) for w in self.scenario.walls]
                self.current_pillars = [list(p) for p in self.scenario.pillars]
                self.gate_wall_indices = list(getattr(self.scenario, "gate_wall_indices", []))
                self._reset_gates()
                self._rebuild_walls_gpu()
                self._rebuild_flow_field()

            if restart_all:
                self.current_epoch = 1
                self.epoch_wins_per_faction = [0, 0, 0, 0]
                for gen in self.brains.generals:
                    gen.xp = 0.0
                    gen.level = 0
                    gen.class_roi_ema = [0.0, 0.0, 0.0, 0.0]
                    gen.composition_weights = torch.tensor([1.0, -10.0, -10.0, -10.0], dtype=torch.float32, device=self.device)
                self.brains.init_population(0, torch.zeros((0,), device=self.device))

            scen_name = SCENARIO_NAMES[self.current_scenario_idx]
            self.combat_events = [{"time": 0.0, "text": f"OPERACE #{self.current_epoch:02d} // SEKTOR: {scen_name}", "team": -1}]

            # ==============================================================
            # NÁKUP JEDNOTEK KAŽDÉ FRAKCE A PŘIŘAZENÍ TŘÍDNĚ DEDIKOVANÝCH SLOTŮ
            # ==============================================================
            all_teams = []
            all_classes = []
            all_brain_slots = []
            self.initial_team_counts = []

            self.faction_active_techs = [[] for _ in range(self.n_factions)]
            for f in range(self.n_factions):
                gen = self.brains.generals[f]
                gen.update_level()
                # Dvoufázový strategický nákup: Jednotky + Power-Ups
                army_classes, bought_techs = gen.allocate_full_budget(total_budget=self.budget_per_faction)
                self.faction_active_techs[f] = bought_techs

                f_len = min(len(army_classes), self.brains.SLOTS_PER_FACTION)
                army_classes = army_classes[:f_len]

                self.initial_team_counts.append(f_len)
                all_teams.extend([f] * f_len)
                all_classes.extend(army_classes)

                # Třídně specializované sloty (Sniper se nikdy nekříží s Pěšákem)
                f_slots = self.brains.get_class_slots_for_army(f, army_classes)
                all_brain_slots.append(f_slots)

            self.N = len(all_teams)
            self.brain_slots = torch.cat(all_brain_slots, dim=0)
            self.teams = torch.tensor(all_teams, dtype=torch.int8, device=self.device)
            self.classes = torch.tensor(all_classes, dtype=torch.int8, device=self.device)

            # Vyvážené atributy: Pěšák (1b), Medik (2b), Sniper (3b), Juggernaut (4b s Heavy Armor)
            self.class_max_hp = torch.tensor([115.0, 105.0, 70.0, 480.0], device=self.device)[self.classes.long()]
            self.class_speed = torch.tensor([3.20, 3.40, 2.20, 2.50], device=self.device)[self.classes.long()]
            self.class_range = torch.tensor([240.0, 200.0, 340.0, 220.0], device=self.device)[self.classes.long()]
            self.class_dmg = torch.tensor([28.0, 18.0, 48.0, 36.0], device=self.device)[self.classes.long()]
            self.class_mag_size = torch.tensor([32.0, 24.0, 6.0, 90.0], device=self.device)[self.classes.long()]

            # Aplikace perků vrchního velení (Generál Level 4+)
            for f in range(self.n_factions):
                f_lvl = getattr(self.brains.generals[f], 'level', 0)
                f_mask = self.teams == f
                if not f_mask.any():
                    continue
                # Level 5: Logistické zásobování (+15 % kapacita zásobníku)
                if f_lvl >= 5:
                    self.class_mag_size[f_mask] = torch.round(self.class_mag_size[f_mask] * 1.15)
                # Level 9: Těžká protekce (+15 % HP Juggernauta)
                if f_lvl >= 9:
                    is_jugg = f_mask & (self.classes == CLASS_JUGGERNAUT)
                    self.class_max_hp[is_jugg] = torch.round(self.class_max_hp[is_jugg] * 1.15)
                # Level 10: Kombinované vojsko (synergie 4 tříd dává +8 % poškození)
                if f_lvl >= 10:
                    unique_c = torch.unique(self.classes[f_mask])
                    if unique_c.numel() == 4:
                        self.class_dmg[f_mask] = self.class_dmg[f_mask] * 1.08

            self.pos = torch.zeros((self.N, 2), dtype=torch.float32, device=self.device)
            self.vel = torch.zeros((self.N, 2), dtype=torch.float32, device=self.device)
            self.heading = torch.zeros((self.N,), dtype=torch.float32, device=self.device)
            self.spawn_centers = torch.zeros((4, 2), dtype=torch.float32, device=self.device)

            for f in range(self.n_factions):
                matching_bases = [b for b in self.scenario.bases if b.get("team") == f]
                base_info = matching_bases[0] if matching_bases else self.scenario.bases[f % len(self.scenario.bases)]
                cx, cy = base_info["x"], base_info["y"]
                self.spawn_centers[f, 0] = cx
                self.spawn_centers[f, 1] = cy

                f_mask = self.teams == f
                c_cnt = int(f_mask.sum().item())
                if c_cnt > 0:
                    self.pos[f_mask, 0] = (torch.randn(c_cnt, device=self.device) * 32.0 + cx).clamp(40.0, self.W - 40.0)
                    self.pos[f_mask, 1] = (torch.randn(c_cnt, device=self.device) * 32.0 + cy).clamp(40.0, self.H - 40.0)
                    self.heading[f_mask] = math.atan2(self.H * 0.5 - cy, self.W * 0.5 - cx)

            self.pos, self.vel = self.spatial.resolve_obstacle_collisions(self.pos, self.vel, self.walls_gpu, self.pillars_gpu, 10.0)

            self.hp = self.class_max_hp.clone()
            self.ammo_in_mag = self.class_mag_size.clone()
            # Kinetické štíty (TECH_SHIELDS: 100% absorpce 1. zásahu)
            self.has_kinetic_shield = torch.zeros((self.N,), dtype=torch.bool, device=self.device)
            for f in range(self.n_factions):
                if 2 in self.faction_active_techs[f]: # TECH_SHIELDS = 2
                    self.has_kinetic_shield[self.teams == f] = True
                if 3 in self.faction_active_techs[f]: # TECH_OVERCLOCK = 3 (+25% kapacita zásobníku)
                    self.class_mag_size[self.teams == f] = torch.round(self.class_mag_size[self.teams == f] * 1.25)
                    self.ammo_in_mag[self.teams == f] = self.class_mag_size[self.teams == f].clone()
            self.reload_timer = torch.zeros((self.N,), dtype=torch.float32, device=self.device)
            self.cooldown = torch.zeros((self.N,), dtype=torch.int32, device=self.device)
            self.suppression = torch.zeros((self.N,), dtype=torch.float32, device=self.device)
            self.in_cover = torch.zeros((self.N,), dtype=torch.bool, device=self.device)
            self.incoming_fire_dir = torch.zeros((self.N, 2), dtype=torch.float32, device=self.device)
            self.has_los_target = torch.zeros((self.N,), dtype=torch.bool, device=self.device)
            self.last_target_enemy = torch.full((self.N,), -1, dtype=torch.long, device=self.device)
            self.fitness = torch.zeros((self.N,), dtype=torch.float32, device=self.device)
            self.damage_dealt = torch.zeros((self.N,), dtype=torch.float32, device=self.device)
            self.damage_taken = torch.zeros((self.N,), dtype=torch.float32, device=self.device)
            self.healing_done = torch.zeros((self.N,), dtype=torch.float32, device=self.device)
            self.unit_kills = torch.zeros((self.N,), dtype=torch.int32, device=self.device)
            self.last_sensors = torch.zeros((self.N, 14), dtype=torch.float32, device=self.device)
            self.sniper_camp_timer = torch.zeros((self.N,), dtype=torch.float32, device=self.device)
            self.stagger_timer = torch.zeros((self.N,), dtype=torch.float32, device=self.device)
            self.ability_cooldown = torch.zeros((self.N,), dtype=torch.float32, device=self.device)
            self.fortress_shield_timer = torch.zeros((self.N,), dtype=torch.float32, device=self.device)
            self.sniper_focus_timer = torch.zeros((self.N,), dtype=torch.float32, device=self.device)
            self.squad_ping = torch.zeros((self.N,), dtype=torch.float32, device=self.device)
            self.unit_state = torch.zeros((self.N,), dtype=torch.int32, device=self.device)
            self.active_smokes_gpu = torch.zeros((0, 5), dtype=torch.float32, device=self.device)  # [x, y, radius, timer, team]
            self._cached_cover_dir = torch.zeros((self.N, 2), dtype=torch.float32, device=self.device)
            self._cached_target_enemy = torch.full((self.N,), -1, dtype=torch.long, device=self.device)
            self._cached_dist_enemy = torch.full((self.N,), 9999.0, dtype=torch.float32, device=self.device)
            self._cached_dir_enemy = torch.zeros((self.N, 2), dtype=torch.float32, device=self.device)
            if not hasattr(self, "staging_buf") or self.staging_buf.num_units != self.N:
                self.staging_buf = SimulationStagingBuffer(self.N, self.device)
            else:
                self.staging_buf.reset()

            if self.nodes_gpu.shape[0] > 0:
                if self.current_scenario_idx == SCENARIO_CITADEL:
                    self.nodes_gpu[:, 3] = -1.0
                    self.nodes_gpu[:, 4] = 0.0
                    self.nodes_gpu[0, 3] = 0.0
                    self.nodes_gpu[0, 4] = 1.0
                else:
                    self.nodes_gpu[:, 3] = -1.0
                    self.nodes_gpu[:, 4] = 0.0

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
        with self._engine_lock:
            alive = self.hp > 0.0
            if self.epoch_winner < 0:
                scores = []
                for f in range(self.n_factions):
                    alive_c = int(((self.teams == f) & alive).sum().item())
                    kills_c = int(self.kills_per_faction[f].item())
                    pts_c = float(self.node_control_points[f].item())
                    scores.append(kills_c * 3.0 + pts_c * 2.5 + alive_c)
                self.epoch_winner = int(np.argmax(scores))

            if self.epoch_winner >= 0:
                self.epoch_wins_per_faction[self.epoch_winner] += 1
                self.brains.generals[self.epoch_winner].total_campaign_wins += 1

            # Bonus za přežití
            self.fitness = torch.where(alive, self.fitness + 18.0, self.fitness)

            # Udělení XP generálům
            for f in range(self.n_factions):
                f_mask = self.teams == f
                f_kills = self.kills_per_faction[f].item()
                f_nodes = self.node_control_points[f].item()
                f_surv = (f_mask & alive).sum().item()
                earned_xp = f_kills * 4.0 + f_nodes * 2.5 + f_surv * 1.0
                if f == self.epoch_winner:
                    earned_xp += 40.0
                self.brains.generals[f].xp += earned_xp

            # Výpočet globálního ROI reportu podle tříd jednotek
            class_roi_report = []
            for c in range(4):
                c_mask = self.classes == c
                cnt = int(c_mask.sum().item())
                if cnt > 0:
                    avg_fit = float(torch.mean(self.fitness[c_mask]).item())
                    tot_dmg = float(torch.sum(self.damage_dealt[c_mask]).item())
                    tot_dmg_taken = float(torch.sum(self.damage_taken[c_mask]).item())
                    tot_heal = float(torch.sum(self.healing_done[c_mask]).item())
                    tot_kills = int(torch.sum(self.unit_kills[c_mask]).item())
                    surv_rate = float((c_mask & alive).sum().item()) / cnt
                    roi_val = avg_fit / float(CLASS_COSTS[c])
                    class_roi_report.append({
                        "class_id": c,
                        "class_name": CLASS_NAMES[c],
                        "cost": CLASS_COSTS[c],
                        "count": cnt,
                        "avg_fitness": round(avg_fit, 2),
                        "roi_per_point": round(roi_val, 2),
                        "total_damage": round(tot_dmg, 1),
                        "total_damage_taken": round(tot_dmg_taken, 1),
                        "total_healing": round(tot_heal, 1),
                        "total_kills": tot_kills,
                        "survival_rate": round(surv_rate, 3),
                    })

            if self.telemetry_mgr is not None:
                factions_stats = []
                for f in range(self.n_factions):
                    f_mask = self.teams == f
                    f_alive = (f_mask & alive).sum().item()
                    f_fit = self.fitness[f_mask]
                    gen = self.brains.generals[f]
                    factions_stats.append({
                        "faction_id": f,
                        "level": gen.level,
                        "xp": float(gen.xp),
                        "alive_count": int(f_alive),
                        "kills": int(self.kills_per_faction[f].item()),
                        "node_points": float(self.node_control_points[f].item()),
                        "fitness_max": float(torch.max(f_fit).item()) if f_fit.numel() > 0 else 0.0,
                        "fitness_avg": float(torch.mean(f_fit).item()) if f_fit.numel() > 0 else 0.0,
                        "class_probs": [round(float(p), 3) for p in gen.get_class_probabilities()],
                    })

                self.telemetry_mgr.log_epoch(
                    epoch_num=self.current_epoch,
                    scenario_name=self.scenario.name,
                    duration_ticks=int(self.operation_time_sec * 60),
                    winner_team=self.epoch_winner,
                    end_reason=self.epoch_end_reason or "OPERAČNÍ VÍTĚZSTVÍ",
                    factions_stats=factions_stats,
                    total_wins=list(self.epoch_wins_per_faction),
                    class_roi=class_roi_report,
                )

            # Třídně specializovaná evoluce mozků
            self.brains.evolve_epoch(
                brain_slots=self.brain_slots,
                teams=self.teams,
                classes=self.classes,
                fitness=self.fitness,
                n_factions=self.n_factions,
                winner_team=self.epoch_winner,
            )
            self.current_epoch = self.brains.epoch_number

            next_scenario = (self.current_scenario_idx + 1) % 3
            self.switch_scenario(next_scenario)

    def _compute_general_orders(self, dir_enemy: torch.Tensor, dist_enemy: torch.Tensor) -> torch.Tensor:
        """Vektorové rozkazy generála podle doktríny a stavu uzlů (plně vektorizováno bez Python cyklů)."""
        with self._engine_lock:
            N = self.N
            if self.nodes_gpu.shape[0] == 0:
                return dir_enemy.clone()

            nodes_xy = self.nodes_gpu[:, 0:2]
            node_owners = self.nodes_gpu[:, 3]
            node_d = torch.cdist(self.pos, nodes_xy)  # (N, num_nodes)
            num_nodes = node_d.shape[1]

            teams_long = self.teams.long()
            owners_exp = node_owners.unsqueeze(0).expand(N, num_nodes)
            teams_exp = teams_long.unsqueeze(1).expand(N, num_nodes)
            is_owned = owners_exp == teams_exp
            is_unowned = ~is_owned

            # Doktríny frakcí
            doctrines = self.faction_doctrine_tensor[teams_long]
            bias = is_unowned.float() * 140.0
            rush_bias = is_unowned.float() * 220.0
            rush_bias[:, 0] += 120.0
            hold_bias = is_owned.float() * 250.0
            skirmish_bias = torch.zeros_like(bias)
            if num_nodes > 1:
                skirmish_bias[:, 0] -= 180.0

            doc_bias = torch.where(
                doctrines.unsqueeze(1) == DOCTRINE_RUSH, rush_bias,
                torch.where(
                    doctrines.unsqueeze(1) == DOCTRINE_HOLD, hold_bias,
                    torch.where(doctrines.unsqueeze(1) == DOCTRINE_SKIRMISH, skirmish_bias, bias)
                )
            )

            eff_d = node_d - doc_bias
            target_n = torch.argmin(eff_d, dim=1)
            vec = nodes_xy[target_n] - self.pos
            norm = torch.norm(vec, dim=1, keepdim=True).clamp(min=1.0)
            return vec / norm

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
        self.in_cover = torch.norm(wall_force, dim=1) > 0.38

        # Taktická odměna za setrvání v krytu pod palbou
        under_fire_in_cover = alive & self.in_cover & (self.suppression > 0.25)
        self.fitness = torch.where(under_fire_in_cover, self.fitness + 0.15, self.fitness)

        # Fáze 1: Detekce nepřátel/spojenců, léčení Medika, egocentrické senzory a forward mozku
        run_sensors_and_brain(self, ctx)
        dir_enemy = ctx.dir_enemy
        dist_enemy = ctx.dist_enemy
        target_enemy = ctx.target_enemy
        medic_has_patient = ctx.medic_has_patient
        p_alive = ctx.p_alive
        t_alive = ctx.t_alive
        dist_mat = ctx.dist_mat
        diff_all = ctx.diff_all
        same_team = ctx.same_team
        cos_th = ctx.cos_th
        sin_th = ctx.sin_th
        brain_out = ctx.brain_out

        # ==============================================================
        # REÁLNÉ POHYBOVÉ ŘÍZENÍ Z AKCÍ MOZKU (Žádné skriptové berličky)
        # Integrace pohybu probíhá PŘED generováním vizuálních stop, aby
        # tracers/lasery vždy vycházely z aktuální pozice vojáka.
        # ==============================================================
        squad_force = torch.zeros((N, 2), dtype=torch.float32, device=self.device)
        if K > 1:
            ally_mask = same_team & (dist_mat > 12.0) & (dist_mat < 145.0)
            ally_weights = torch.where(ally_mask, 1.0 / dist_mat.clamp(min=12.0), torch.zeros_like(dist_mat))
            weight_sum = ally_weights.sum(dim=1, keepdim=True).clamp(min=1e-5)
            squad_diff = torch.bmm(ally_weights.unsqueeze(1), diff_all).squeeze(1) / weight_sum
            squad_force[alive_idx] = -squad_diff * 0.35

        # ==============================================================
        # 7 Tenzorových Výstupů Mozku:
        # 0: Surge, 1: Sway, 2: Yaw, 3: Trigger, 4: Stance, 5: Ability, 6: Ping
        # ==============================================================
        brain_surge  = brain_out[:, 0]  # +1 vpřed, -1 plná zpátečka / couvání
        brain_sway   = brain_out[:, 1]  # +1 vpravo, -1 vlevo
        brain_yaw    = brain_out[:, 2]  # rotace těla nezávisle na směru pohybu
        brain_stance = brain_out[:, 4]  # +1 Sprint (+30% rychlost, horší kryt), -1 Plížení (poloviční rychlost, max kryt)
        brain_ability= brain_out[:, 5]  # > 0.60 spouští aktivní třídní schopnost
        brain_ping   = brain_out[:, 6]  # taktický signál vysílaný spojencům v četě
    
        # Aktualizace komunikačního kanálu čety
        self.squad_ping = torch.lerp(self.squad_ping, brain_ping * alive.float(), 0.35)

        # Vyhodnocení třídních schopností (Akce 5: Ability) přes CombatSystem
        (
            self.ability_cooldown,
            self.active_smokes_gpu,
            self.fortress_shield_timer,
            self.sniper_focus_timer,
        ) = self.combat.process_abilities(
            dt=dt,
            brain_ability=brain_ability,
            alive=alive,
            classes=self.classes,
            teams=self.teams,
            pos=self.pos,
            hp=self.hp,
            class_max_hp=self.class_max_hp,
            ability_cooldown=self.ability_cooldown,
            active_smokes_gpu=self.active_smokes_gpu,
            fortress_shield_timer=self.fortress_shield_timer,
            sniper_focus_timer=self.sniper_focus_timer,
            staging_buf=self.staging_buf,
            healing_done=self.healing_done,
            fitness=self.fitness,
            active_heals_p1=[],
            active_heals_p2=[],
            faction_techs=self.faction_active_techs,
            visual_mode=self.visual_mode,
            particles=self.particles,
        )

        # Holonomická transformace tělesného vektoru (Surge & Sway) do globálních souřadnic
        # Agent se může pohybovat libovolným směrem v 360° a zároveň mířit jinam!
        brain_mv_x = brain_surge * cos_th + brain_sway * (-sin_th)
        brain_mv_y = brain_surge * sin_th + brain_sway * cos_th
        brain_world_vec = torch.stack([brain_mv_x, brain_mv_y], dim=1)

        # Spojení tahu s odpudivostí překážek a kohezí
        move_raw = brain_world_vec * 1.35 + squad_force * 0.18 + wall_force * 1.60

        # Odpočet Stagger timeru (ochrnutí hybnosti po těžkém zásahu)
        self.stagger_timer = (self.stagger_timer - dt).clamp(min=0.0)
        is_staggered = self.stagger_timer > 0.0

        # ==============================================================
        # MATICOVÝ STAVOVÝ REGISTR (ArgMax State Selector)
        # Tenzorové vyhodnocení stavů bez vnořených větví if/else
        # Stavy: 0: IDLE, 1: ENGAGE, 2: COVER_RETREAT, 3: SNIPER_LOCK, 4: TRIAGE_HEAL, 5: FORTRESS_PUSH
        # ==============================================================
        utility = torch.zeros((N, 6), dtype=torch.float32, device=self.device)
        u_classes = self.classes.long()

        # Stav 0: IDLE (výchozí základní váha)
        utility[:, STATE_IDLE] = 0.50

        # Stav 1: ENGAGE (čelní cíl v dosahu, nízké potlačení)
        has_combat_target = (dist_enemy <= self.class_range * 1.35) & self.has_los_target & (target_enemy >= 0) & alive
        enemy_ang = torch.atan2(dir_enemy[:, 1], dir_enemy[:, 0])
        aim_diff = (enemy_ang - self.heading + math.pi) % (2.0 * math.pi) - math.pi
        engage_utility = torch.where(
            has_combat_target & (torch.abs(aim_diff) < math.radians(45)),
            1.20 + (1.0 - self.suppression) * 0.80,
            0.0,
        )
        utility[:, STATE_ENGAGE] = engage_utility

        # Stav 2: COVER_RETREAT (těžké potlačení nebo poškození mimo kryt -> pud sebezáchovy)
        retreat_utility = torch.where(
            (~self.in_cover) & ((self.suppression > 0.40) | (self.hp < self.class_max_hp * 0.35)),
            1.60 + self.suppression * 0.90,
            0.0,
        )
        utility[:, STATE_COVER_RETREAT] = retreat_utility

        # Stav 3: SNIPER_LOCK (Sniper s čistým LoS raycastem zastavuje pohyb a fixuje zbraň)
        is_sniper_cls = u_classes == CLASS_SNIPER
        sniper_lock_utility = torch.where(
            is_sniper_cls & has_combat_target & (self.ammo_in_mag > 0),
            2.80,
            0.0,
        )
        utility[:, STATE_SNIPER_LOCK] = sniper_lock_utility

        # Stav 4: TRIAGE_HEAL (Medik s pacientem v dosahu ignoruje útok)
        is_medic_cls = u_classes == CLASS_MEDIC
        medic_heal_utility = torch.where(
            is_medic_cls & medic_has_patient,
            3.00,
            0.0,
        )
        utility[:, STATE_TRIAGE_HEAL] = medic_heal_utility

        # Stav 5: FORTRESS_PUSH (Juggernaut neustupuje, ignoruje lehký stagger a tlačí frontu)
        is_jugg_cls = u_classes == CLASS_JUGGERNAUT
        jugg_push_utility = torch.where(
            is_jugg_cls & alive,
            1.90,
            0.0,
        )
        utility[:, STATE_FORTRESS_PUSH] = jugg_push_utility

        # Čistý tenzorový ArgMax výběr stavu jednotky (pouze pro telemetrickou diagnostiku v HUDu, bez override pohybu/záměru):
        self.unit_state = torch.argmax(utility, dim=1).int()

        # Zero-Script AI: Pohyb je plně řízen výstupem neuronové sítě (Surge, Sway, Stance)
        # s přihlédnutím k fyzikálnímu odpuzování od stěn a kohezi formace.
        effective_move_raw = move_raw

        u_doc = self.faction_doctrine_tensor[self.teams.long()]
        doc_speed_mod = torch.where(u_doc == DOCTRINE_RUSH, 1.12, torch.where(u_doc == DOCTRINE_HOLD, 0.90, 1.0))
        # Stance modifikátor rychlosti: +1.0 = Sprint (1.30x), -1.0 = Plížení (0.55x)
        stance_speed_mod = torch.where(brain_stance > 0.3, 1.30, torch.where(brain_stance < -0.3, 0.55, 1.0))
        speed_mult = torch.where(is_staggered, torch.full_like(doc_speed_mod, 0.15), (1.0 - self.suppression * 0.40) * doc_speed_mod * stance_speed_mod)

        v_norm = torch.norm(effective_move_raw, dim=1, keepdim=True).clamp(min=1.0)
        desired_vel = (effective_move_raw / v_norm) * speed_mult.unsqueeze(1) * self.class_speed.unsqueeze(1)

        self.vel = torch.lerp(self.vel, desired_vel, 0.26) * alive.unsqueeze(1).float()
        self.pos += self.vel

        self.pos, self.vel = self.spatial.resolve_obstacle_collisions(self.pos, self.vel, self.walls_gpu, self.pillars_gpu, 10.0)
        self.pos[:, 0].clamp_(25.0, self.W - 25.0)
        self.pos[:, 1].clamp_(25.0, self.H - 25.0)

        # Nezávislé otáčení těla a zaměřování (Yaw):
        # Rotace je řízena mozkem (brain_yaw). V boji s přímou viditelností poskytuje jemnou taktickou asistenci.
        omega = torch.where(has_combat_target, aim_diff * 0.45 + brain_yaw * 0.20, brain_yaw * 0.45)
        self.heading = (self.heading + omega * alive.float() + math.pi) % (2.0 * math.pi) - math.pi

        # ANTI-KÁČA FITNESS PENALIZACE ZA ZBYTEČNOU ÚHLOVOU KINETIKU:
        # Pokud se agent bezcílně točí na místě bez lineárního pohybu, okamžitě padá jeho fitness!
        lin_speed = torch.norm(self.vel, dim=1)
        spin_penalty = torch.where((lin_speed < 0.20) & (~has_combat_target) & alive, omega.abs() * 0.18, torch.zeros_like(omega))
        self.fitness -= spin_penalty

        # Okamžitá ztráta zaměření cíle mimo FOV kužel (100 % bezvětveně na GPU bez .any()/nonzero)
        has_tar = (target_enemy >= 0) & (target_enemy < self.N) & alive
        tar_safe = target_enemy.clamp(0, max(0, N - 1))
        tar_v = self.pos[tar_safe] - self.pos
        tar_d = torch.norm(tar_v, dim=1)
        tar_a = torch.atan2(tar_v[:, 1], tar_v[:, 0])
        tar_diff = (tar_a - self.heading + math.pi) % (2.0 * math.pi) - math.pi
        lost_fov = has_tar & (tar_d > 75.0) & (torch.abs(tar_diff) >= (math.radians(140) * 0.5))

        target_enemy = torch.where(lost_fov, -1, target_enemy)
        dist_enemy = torch.where(lost_fov, 9999.0, dist_enemy)
        dir_enemy = torch.where(lost_fov.unsqueeze(1), 0.0, dir_enemy)
        self.has_los_target = self.has_los_target & (~lost_fov)

        self.last_target_enemy = target_enemy

        # DETERMINISTICKÝ AUTOMAT PŘEBÍJENÍ (Bez zbytečného zatěžování mozku přebíjením):
        empty_mag = self.ammo_in_mag <= 0
        tactical_cover_reload = self.in_cover & (self.ammo_in_mag < self.class_mag_size * 0.45) & (~has_combat_target)
        start_reload = alive & (empty_mag | tactical_cover_reload) & (self.reload_timer <= 0.0) & (self.ammo_in_mag < self.class_mag_size)
        self.reload_timer = torch.where(start_reload, self.CONST_RELOAD_TIME, self.reload_timer)

        # Střelba: Výstup 3 je Trigger (Sigmoid [0, 1] > 0.58)
        self.cooldown = (self.cooldown - 1).clamp(min=0)

        valid_tar = (target_enemy >= 0) & (target_enemy < self.N)
        # tar_diff už máme spočítaný ze stejných pozic a headingů výše -> 0 duplicitních atan2/nonzero!
        on_target = alive & valid_tar & (torch.abs(tar_diff) < math.radians(45))

        # Trigger je nyní čistý Sigmoid práh > 0.58 z výstupu 3
        brain_trigger = brain_out[:, 3] > 0.58

        can_fire = (
            alive &
            valid_tar &
            self.has_los_target &
            on_target &
            brain_trigger &
            (self.ammo_in_mag > 0) &
            (self.reload_timer <= 0.0) &
            (self.cooldown == 0) &
            (dist_enemy <= self.class_range)
        )
        shooters = torch.where(can_fire)[0]

        self.active_tracers_p1 = self._empty_xy
        self.active_tracers_p2 = self._empty_xy
        self.active_tracers_team = self._empty_i8
        self.hit_sparks = self._empty_xy
        self.armor_sparks = self._empty_xy

        if shooters.numel() > 0:
            targets = target_enemy[shooters]
            p_from = self.pos[shooters]
            p_to = self.pos[targets]

            los_ok = check_line_of_sight(p_from, p_to, self.pillars_gpu, self.walls_gpu, self.active_smokes_gpu)
            valid_s = shooters[los_ok]
            valid_t = targets[los_ok]

            if valid_s.numel() > 0:
                self.ammo_in_mag[valid_s] -= 1.0
                s_classes = self.classes[valid_s].long()
                cd_val = self.CONST_CLASS_COOLDOWNS[s_classes]
                self.cooldown[valid_s] = cd_val

                fire_vec = self.pos[valid_s] - self.pos[valid_t]
                fire_norm = torch.norm(fire_vec, dim=1, keepdim=True).clamp(min=1.0)
                self.incoming_fire_dir[valid_t] = fire_vec / fire_norm

                # Přesnost podle třídy
                base_acc = self.CONST_CLASS_BASE_ACC[s_classes]
                cover_mult = torch.where(s_classes == CLASS_SNIPER, 0.60, 0.48)
                t_doc = self.faction_doctrine_tensor[self.teams[valid_t].long()]
                hold_bonus = torch.where(t_doc == DOCTRINE_HOLD, 0.85, 1.0)

                hit_prob = torch.where(self.in_cover[valid_t], base_acc * cover_mult * hold_bonus, base_acc)
                # Sniper Focus buff: +0.20 přesnost pro snipry s aktivním fokusem
                sniper_focus_active = self.sniper_focus_timer[valid_s] > 0.0
                hit_prob = torch.where(sniper_focus_active, (hit_prob + 0.20).clamp(max=0.98), hit_prob)
                hit_prob = hit_prob * (1.0 - self.suppression[valid_s] * 0.45)

                is_hit = torch.rand(valid_s.numel(), device=self.device) < hit_prob
            
                # ==============================================================
                # AEGIS 2.0 VEKTOROVÁ BALISTIKA: Úhlový pancíř, Zónový rozpad & Kinetický Ragdoll
                # ==============================================================
                b_res = self.ballistics.resolve_ballistic_hits(
                    valid_s=valid_s,
                    valid_t=valid_t,
                    is_hit=is_hit,
                    shooter_pos=self.pos[valid_s],
                    target_pos=self.pos[valid_t],
                    target_heading=self.heading[valid_t],
                    shooter_classes=self.classes[valid_s],
                    target_classes=self.classes[valid_t],
                    target_in_cover=self.in_cover[valid_t],
                    has_kinetic_shield=self.has_kinetic_shield if hasattr(self, "has_kinetic_shield") else torch.zeros_like(self.hp, dtype=torch.bool),
                    base_dmg=self.class_dmg[valid_s],
                    sniper_focus_active=self.sniper_focus_timer[valid_s] > 0.0,
                    fortress_shield_active=self.fortress_shield_timer[valid_t] > 0.0,
                )

                dmg = b_res["final_damage"]
                # Prasknutí kinetických štítů při absorpci
                if hasattr(self, "has_kinetic_shield") and b_res["shield_absorbed"].any():
                    self.has_kinetic_shield[valid_t[b_res["shield_absorbed"]]] = False

                # Vizuální jiskry odraženého pancíře (Armor Sparks)
                if self.visual_mode and b_res["flank_hit"].any():
                    self.armor_sparks = self.pos[valid_t[b_res["flank_hit"]][:24]]

                # ==============================================================
                # ZERO-TRUST WRITE-ONLY BUFFER: Všechny impulzy a delty se ukládají zde!
                # ==============================================================
                # Kinetický impuls (Knockback)
                kb_vectors = torch.stack([b_res["knockback_x"], b_res["knockback_y"]], dim=1)
                self.staging_buf.delta_vel.index_add_(0, valid_t, kb_vectors)

                # Poškození (záporné HP) a statistiky zásahu
                self.staging_buf.delta_hp.index_add_(0, valid_t, -dmg)
                self.staging_buf.damage_taken.index_add_(0, valid_t, dmg)
            
                # Záznam střelce pro přesnou atribuci killů
                if is_hit.any():
                    hit_t = valid_t[is_hit]
                    hit_s = valid_s[is_hit]
                    self.staging_buf.last_attacker_id[hit_t] = hit_s

                # Suppression yield
                self.staging_buf.delta_suppression.index_add_(0, valid_t, b_res["suppression_yield"])

                # Motoric Stagger duration (zapíše se nejvyšší hodnota trvání)
                if b_res["stagger_mask"].any():
                    stagg_idx = valid_t[b_res["stagger_mask"]]
                    stagg_dur = b_res["stagger_duration"][b_res["stagger_mask"]]
                    self.staging_buf.stagger_duration[stagg_idx] = torch.maximum(
                        self.staging_buf.stagger_duration[stagg_idx], stagg_dur
                    )

                # Pud sebezáchovy: penalizace za poškození utržené mimo kryt
                out_of_cover_hit = is_hit & (~self.in_cover[valid_t])
                if out_of_cover_hit.any():
                    pen_targets = valid_t[out_of_cover_hit]
                    pen_dmg = (dmg * 0.22)[out_of_cover_hit]
                    self.staging_buf.out_of_cover_penalty.index_add_(
                        0, pen_targets, pen_dmg
                    )

                # Near-miss suppression pro jednotky v okolí zásahu
                if K > 1 and valid_t.numel() > 0:
                    d_to_targets = torch.cdist(p_alive, self.pos[valid_t])
                    near_miss = (d_to_targets < 65.0).any(dim=1)
                    if near_miss.any():
                        supp_targets = alive_idx[near_miss]
                        supp_vals = torch.full((supp_targets.numel(),), 0.10, device=self.device)
                        self.staging_buf.delta_suppression.index_add_(
                            0, supp_targets, supp_vals
                        )

                self.damage_dealt[valid_s] += dmg
                self.fitness[valid_s] += dmg * 0.65

                if self.visual_mode:
                    s_lim = min(valid_s.numel(), 64)
                    self.active_tracers_p1 = self.pos[valid_s[:s_lim]]
                    self.active_tracers_p2 = self.pos[valid_t[:s_lim]]
                    self.active_tracers_team = self.teams[valid_s[:s_lim]]
                    if is_hit.any():
                        self.hit_sparks = self.pos[valid_t[is_hit][:32]]
                        is_target_jugg = self.classes[valid_t].long() == CLASS_JUGGERNAUT
                        jugg_hits = is_hit & is_target_jugg
                        if jugg_hits.any():
                            self.armor_sparks = self.pos[valid_t[jugg_hits][:24]]

        # ==============================================================
        # FÁZE 3: ATOMICKÝ HROMADNÝ COMMIT (Zero-Trust Pipeline)
        # Všechny akumulované delty ze staging bufferu aplikovány v jediném průchodu!
        # ==============================================================
        self._atomic_commit_pipeline(dt)

        self.last_target_enemy = target_enemy

        # Zaměřovací lasery Sniperů – POUZE pokud mají přímou viditelnost a míří na cíl dopředu!
        aiming_snipers = alive & (self.classes == CLASS_SNIPER) & (dist_enemy <= self.class_range) & (dist_enemy > 10.0) & self.has_los_target
        if not self.visual_mode:
            aiming_snipers = torch.zeros_like(aiming_snipers)
        aim_idx = torch.where(aiming_snipers)[0]
        if aim_idx.numel() > 0:
            a_tar = target_enemy[aim_idx]
            valid_tar_snipers = a_tar >= 0
            if valid_tar_snipers.any():
                s_valid = aim_idx[valid_tar_snipers]
                t_valid = a_tar[valid_tar_snipers]
                l_vec = self.pos[t_valid] - self.pos[s_valid]
                l_ang = torch.atan2(l_vec[:, 1], l_vec[:, 0])
                l_err = (l_ang - self.heading[s_valid] + math.pi) % (2.0 * math.pi) - math.pi
                facing = torch.abs(l_err) < math.radians(50)
                if facing.any():
                    s_face = s_valid[facing]
                    t_face = t_valid[facing]
                    a_lim = min(s_face.numel(), 20)
                    a_sel = s_face[:a_lim]
                    a_tar_sel = t_face[:a_lim]
                    self.sniper_lasers = torch.cat([self.pos[a_sel], self.pos[a_tar_sel]], dim=1)
                else:
                    self.sniper_lasers = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
            else:
                self.sniper_lasers = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
        else:
            self.sniper_lasers = torch.zeros((0, 4), dtype=torch.float32, device=self.device)

        # Sniper anti-kemping penalizace (pokud sniper stojí déle než 8s na místě, ztrácí fitness)
        is_sniper = (self.classes == CLASS_SNIPER) & alive
        is_standing = torch.norm(self.vel, dim=1) < 0.20
        self.sniper_camp_timer = torch.where(is_sniper & is_standing, self.sniper_camp_timer + dt, torch.zeros_like(self.sniper_camp_timer))
        camp_penalty = (self.sniper_camp_timer > 8.0) & is_sniper
        if camp_penalty.any():
            self.fitness[camp_penalty] -= 1.8 * dt

        # Aktualizace částic výbuchů a kouře
        alive_particles = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vx"] *= 0.92
            p["vy"] *= 0.92
            p["life"] -= dt * 2.2
            if p["life"] > 0:
                alive_particles.append(p)
        self.particles = alive_particles

        # Obsazování uzlů (plně vektorizovaně na GPU v matici (K, M) bez .item() ve smyčce)
        if self.nodes_gpu.shape[0] > 0 and (self.tick_counter % 6 == 0) and K > 0:
            p_alive = self.pos[alive_idx]
            t_alive = self.teams[alive_idx].long()
            step_scale = 6.0

            # 1. Maticový výpočet přítomnosti jednotek v uzlech: tvar (K, M)
            diff = p_alive.unsqueeze(1) - self.nodes_gpu[:, :2].unsqueeze(0)   # (K, M, 2)
            dists_sq = (diff * diff).sum(dim=-1)                               # (K, M)
            radii_sq = self.nodes_gpu[:, 2].unsqueeze(0) ** 2                  # (1, M)
            in_node = dists_sq <= radii_sq                                     # (K, M) bool

            # 2. Počty jednotek jednotlivých frakcí v každém uzlu: tvar (M, 4)
            team_one_hot = torch.nn.functional.one_hot(t_alive, num_classes=4).to(torch.float32)
            cnts = in_node.to(torch.float32).transpose(0, 1) @ team_one_hot    # (M, 4)

            cnt_dom, dom_t = torch.max(cnts, dim=1)                            # (M,), (M,)
            cnt_other = cnts.sum(dim=1) - cnt_dom                              # (M,)
            has_majority = cnt_dom > cnt_other                                 # (M,) bool

            n_owner = self.nodes_gpu[:, 3].long()                              # (M,)
            n_cap = self.nodes_gpu[:, 4].clone()                               # (M,)

            # 3. Vektorový výpočet fitness pro obsazující jednotky (+250 % bonus pro CLASS_GRUNT)
            is_grunt = self.classes[alive_idx] == CLASS_GRUNT
            bonus_mult = torch.where(is_grunt, 2.5, 1.0).unsqueeze(1)          # (K, 1)
            is_cap_unit = in_node & has_majority.unsqueeze(0) & (t_alive.unsqueeze(1) == dom_t.unsqueeze(0))

            same_owner = n_owner == dom_t                                      # (M,)
            reinforcing = has_majority & same_owner & (n_cap < 1.0)            # (M,)
            neutralizing = has_majority & (~same_owner)                        # (M,)

            node_fit_rate = torch.where(
                reinforcing,
                0.45 * step_scale,
                torch.where(neutralizing, 0.65 * step_scale, 0.0),
            ).unsqueeze(0)                                                     # (1, M)

            cap_fitness_gain = (is_cap_unit.to(torch.float32) * bonus_mult * node_fit_rate).sum(dim=1)
            self.fitness[alive_idx] += cap_fitness_gain

            # 4. Aktualizace progresu obsazení (n_cap) a vlastníka (n_owner) na GPU
            n_cap = torch.where(reinforcing, torch.clamp(n_cap + 0.04 * step_scale, max=1.0), n_cap)
            n_cap = torch.where(neutralizing, n_cap - 0.05 * step_scale, n_cap)

            captured_mask = neutralizing & (n_cap <= 0.0)                      # (M,) bool
            n_owner = torch.where(captured_mask, dom_t, n_owner)
            n_cap = torch.where(captured_mask, torch.full_like(n_cap, 0.12), n_cap)

            self.nodes_gpu[:, 3] = n_owner.to(torch.float32)
            self.nodes_gpu[:, 4] = n_cap

            # 5. Zápis události do combat_events pouze ve vzácném tiku změny vlastníka
            if captured_mask.any():
                cap_indices = torch.where(captured_mask)[0].cpu().tolist()
                dom_teams_cpu = dom_t[captured_mask].cpu().tolist()
                sec_t = int(self.operation_time_sec)
                meta_names = ("ALPHA", "OMEGA", "SOLAR", "PHANTOM")
                node_names = ("UZEL ALFA", "UZEL BRAVO", "UZEL CHARLIE", "UZEL DELTA")

                for idx_node, d_team in zip(cap_indices, dom_teams_cpu):
                    if len(self.combat_events) > 35:
                        self.combat_events.pop(0)
                    n_label = node_names[idx_node] if idx_node < 4 else f"UZEL #{idx_node + 1}"
                    self.combat_events.append({
                        "time": self.operation_time_sec,
                        "text": f"[{sec_t:02d}s] {meta_names[d_team]} obsadil {n_label}!",
                        "team": d_team,
                    })

            # 6. Pasivní bodování a obranná fitness bez CPU větvení (n_owner >= 0 a n_cap >= 0.7)
            held_mask = (n_owner >= 0) & (n_cap >= 0.7)                        # (M,) bool
            safe_owners = torch.clamp(n_owner, min=0)
            pts_add = torch.where(held_mask, 0.08 * step_scale, 0.0).to(self.node_control_points.dtype)
            self.node_control_points.index_add_(0, safe_owners, pts_add)

            is_defending = in_node & held_mask.unsqueeze(0) & (t_alive.unsqueeze(1) == n_owner.unsqueeze(0))
            def_nodes_count = is_defending.sum(dim=1).to(torch.float32)        # (K,)
            self.fitness[alive_idx] += def_nodes_count * (0.25 * step_scale)

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
