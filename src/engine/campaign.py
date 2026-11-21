"""
engine/campaign.py - Řízení kampaně, start/konec operace, evoluce a vektorové rozkazy generála.

Bezstavové funkce pracující nad instancí RealisticBattleEngine.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from ai_brains import CLASS_COSTS, CLASS_JUGGERNAUT
from config import CLASS_NAMES
from engine.staging_buffer import SimulationStagingBuffer
from scenarios import SCENARIO_CITADEL, SCENARIO_NAMES

if TYPE_CHECKING:
    from engine_core import RealisticBattleEngine

DOCTRINE_RUSH = 1
DOCTRINE_HOLD = 2
DOCTRINE_SKIRMISH = 3


def start_new_operation(eng: RealisticBattleEngine, restart_all: bool = False) -> None:
    """Inicializuje novou bitvu/operaci, nakoupí armády a alokuje GPU tenzory jednotek."""
    with eng._engine_lock:
        eng.operation_id = getattr(eng, "operation_id", 0) + 1
        eng.operation_time_sec = 0.0
        eng.tick_counter = 0
        eng.epoch_winner = -1
        eng.epoch_end_reason = ""
        eng.kills_per_faction.zero_()
        eng.node_control_points.zero_()
        eng.particles.clear()

        # Obnova zničitelných bran a dynamických stěn pro novou bitvu
        if hasattr(eng, "scenario"):
            eng.current_walls = [list(w) for w in eng.scenario.walls]
            eng.current_pillars = [list(p) for p in eng.scenario.pillars]
            eng.gate_wall_indices = list(getattr(eng.scenario, "gate_wall_indices", []))
            eng._reset_gates()
            eng._rebuild_walls_gpu()
            eng._rebuild_flow_field()

        if restart_all:
            eng.current_epoch = 1
            eng.epoch_wins_per_faction = [0, 0, 0, 0]
            for gen in eng.brains.generals:
                gen.xp = 0.0
                gen.level = 0
                gen.class_roi_ema = [0.0, 0.0, 0.0, 0.0]
                gen.composition_weights = torch.tensor(
                    [1.0, -10.0, -10.0, -10.0], dtype=torch.float32, device=eng.device
                )
            eng.brains.init_population(0, torch.zeros((0,), device=eng.device))

        scen_name = SCENARIO_NAMES[eng.current_scenario_idx]
        eng.combat_events = [
            {"time": 0.0, "text": f"OPERACE #{eng.current_epoch:02d} // SEKTOR: {scen_name}", "team": -1}
        ]

        # ==============================================================
        # NÁKUP JEDNOTEK KAŽDÉ FRAKCE A PŘIŘAZENÍ TŘÍDNĚ DEDIKOVANÝCH SLOTŮ
        # ==============================================================
        all_teams = []
        all_classes = []
        all_brain_slots = []
        eng.initial_team_counts = []

        eng.faction_active_techs = [[] for _ in range(eng.n_factions)]
        for f in range(eng.n_factions):
            gen = eng.brains.generals[f]
            gen.update_level()
            # Dvoufázový strategický nákup: Jednotky + Power-Ups
            army_classes, bought_techs = gen.allocate_full_budget(total_budget=eng.budget_per_faction)
            eng.faction_active_techs[f] = bought_techs

            f_len = min(len(army_classes), eng.brains.SLOTS_PER_FACTION)
            army_classes = army_classes[:f_len]

            eng.initial_team_counts.append(f_len)
            all_teams.extend([f] * f_len)
            all_classes.extend(army_classes)

            # Třídně specializované sloty (Sniper se nikdy nekříží s Pěšákem)
            f_slots = eng.brains.get_class_slots_for_army(f, army_classes)
            all_brain_slots.append(f_slots)

        eng.N = len(all_teams)
        eng.brain_slots = torch.cat(all_brain_slots, dim=0)
        eng.teams = torch.tensor(all_teams, dtype=torch.int8, device=eng.device)
        eng.classes = torch.tensor(all_classes, dtype=torch.int8, device=eng.device)

        # Vyvážené atributy: Pěšák (1b), Medik (2b), Sniper (3b), Juggernaut (4b s Heavy Armor)
        eng.class_max_hp = torch.tensor([115.0, 105.0, 70.0, 480.0], device=eng.device)[eng.classes.long()]
        eng.class_speed = torch.tensor([3.20, 3.40, 2.20, 2.50], device=eng.device)[eng.classes.long()]
        eng.class_range = torch.tensor([240.0, 200.0, 340.0, 220.0], device=eng.device)[eng.classes.long()]
        eng.class_dmg = torch.tensor([28.0, 18.0, 48.0, 36.0], device=eng.device)[eng.classes.long()]
        eng.class_mag_size = torch.tensor([32.0, 24.0, 6.0, 90.0], device=eng.device)[eng.classes.long()]

        # Aplikace perků vrchního velení (Generál Level 4+)
        for f in range(eng.n_factions):
            f_lvl = getattr(eng.brains.generals[f], "level", 0)
            f_mask = eng.teams == f
            if not f_mask.any():
                continue
            # Level 5: Logistické zásobování (+15 % kapacita zásobníku)
            if f_lvl >= 5:
                eng.class_mag_size[f_mask] = torch.round(eng.class_mag_size[f_mask] * 1.15)
            # Level 9: Těžká protekce (+15 % HP Juggernauta)
            if f_lvl >= 9:
                is_jugg = f_mask & (eng.classes == CLASS_JUGGERNAUT)
                eng.class_max_hp[is_jugg] = torch.round(eng.class_max_hp[is_jugg] * 1.15)
            # Level 10: Kombinované vojsko (synergie 4 tříd dává +8 % poškození)
            if f_lvl >= 10:
                unique_c = torch.unique(eng.classes[f_mask])
                if unique_c.numel() == 4:
                    eng.class_dmg[f_mask] = eng.class_dmg[f_mask] * 1.08

        eng.pos = torch.zeros((eng.N, 2), dtype=torch.float32, device=eng.device)
        eng.vel = torch.zeros((eng.N, 2), dtype=torch.float32, device=eng.device)
        eng.heading = torch.zeros((eng.N,), dtype=torch.float32, device=eng.device)
        eng.spawn_centers = torch.zeros((4, 2), dtype=torch.float32, device=eng.device)

        for f in range(eng.n_factions):
            matching_bases = [b for b in eng.scenario.bases if b.get("team") == f]
            base_info = matching_bases[0] if matching_bases else eng.scenario.bases[f % len(eng.scenario.bases)]
            cx, cy = base_info["x"], base_info["y"]
            eng.spawn_centers[f, 0] = cx
            eng.spawn_centers[f, 1] = cy

            f_mask = eng.teams == f
            c_cnt = int(f_mask.sum().item())
            if c_cnt > 0:
                eng.pos[f_mask, 0] = (torch.randn(c_cnt, device=eng.device) * 32.0 + cx).clamp(40.0, eng.W - 40.0)
                eng.pos[f_mask, 1] = (torch.randn(c_cnt, device=eng.device) * 32.0 + cy).clamp(40.0, eng.H - 40.0)
                eng.heading[f_mask] = math.atan2(eng.H * 0.5 - cy, eng.W * 0.5 - cx)

        eng.pos, eng.vel = eng.spatial.resolve_obstacle_collisions(eng.pos, eng.vel, eng.walls_gpu, eng.pillars_gpu, 10.0)

        eng.hp = eng.class_max_hp.clone()
        eng.ammo_in_mag = eng.class_mag_size.clone()
        # Kinetické štíty (TECH_SHIELDS: 100% absorpce 1. zásahu)
        eng.has_kinetic_shield = torch.zeros((eng.N,), dtype=torch.bool, device=eng.device)
        for f in range(eng.n_factions):
            if 2 in eng.faction_active_techs[f]:  # TECH_SHIELDS = 2
                eng.has_kinetic_shield[eng.teams == f] = True
            if 3 in eng.faction_active_techs[f]:  # TECH_OVERCLOCK = 3 (+25% kapacita zásobníku)
                eng.class_mag_size[eng.teams == f] = torch.round(eng.class_mag_size[eng.teams == f] * 1.25)
                eng.ammo_in_mag[eng.teams == f] = eng.class_mag_size[eng.teams == f].clone()
        eng.reload_timer = torch.zeros((eng.N,), dtype=torch.float32, device=eng.device)
        eng.cooldown = torch.zeros((eng.N,), dtype=torch.int32, device=eng.device)
        eng.suppression = torch.zeros((eng.N,), dtype=torch.float32, device=eng.device)
        eng.in_cover = torch.zeros((eng.N,), dtype=torch.bool, device=eng.device)
        eng.incoming_fire_dir = torch.zeros((eng.N, 2), dtype=torch.float32, device=eng.device)
        eng.has_los_target = torch.zeros((eng.N,), dtype=torch.bool, device=eng.device)
        eng.last_target_enemy = torch.full((eng.N,), -1, dtype=torch.long, device=eng.device)
        eng.fitness = torch.zeros((eng.N,), dtype=torch.float32, device=eng.device)
        eng.damage_dealt = torch.zeros((eng.N,), dtype=torch.float32, device=eng.device)
        eng.damage_taken = torch.zeros((eng.N,), dtype=torch.float32, device=eng.device)
        eng.healing_done = torch.zeros((eng.N,), dtype=torch.float32, device=eng.device)
        eng.unit_kills = torch.zeros((eng.N,), dtype=torch.int32, device=eng.device)
        eng.last_sensors = torch.zeros((eng.N, 14), dtype=torch.float32, device=eng.device)
        eng.sniper_camp_timer = torch.zeros((eng.N,), dtype=torch.float32, device=eng.device)
        eng.stagger_timer = torch.zeros((eng.N,), dtype=torch.float32, device=eng.device)
        eng.ability_cooldown = torch.zeros((eng.N,), dtype=torch.float32, device=eng.device)
        eng.fortress_shield_timer = torch.zeros((eng.N,), dtype=torch.float32, device=eng.device)
        eng.sniper_focus_timer = torch.zeros((eng.N,), dtype=torch.float32, device=eng.device)
        eng.squad_ping = torch.zeros((eng.N,), dtype=torch.float32, device=eng.device)
        eng.unit_state = torch.zeros((eng.N,), dtype=torch.int32, device=eng.device)
        eng.active_smokes_gpu = torch.zeros((0, 5), dtype=torch.float32, device=eng.device)  # [x, y, radius, timer, team]
        eng._cached_cover_dir = torch.zeros((eng.N, 2), dtype=torch.float32, device=eng.device)
        eng._cached_target_enemy = torch.full((eng.N,), -1, dtype=torch.long, device=eng.device)
        eng._cached_dist_enemy = torch.full((eng.N,), 9999.0, dtype=torch.float32, device=eng.device)
        eng._cached_dir_enemy = torch.zeros((eng.N, 2), dtype=torch.float32, device=eng.device)
        if not hasattr(eng, "staging_buf") or eng.staging_buf.num_units != eng.N:
            eng.staging_buf = SimulationStagingBuffer(eng.N, eng.device)
        else:
            eng.staging_buf.reset()

        if eng.nodes_gpu.shape[0] > 0:
            if eng.current_scenario_idx == SCENARIO_CITADEL:
                eng.nodes_gpu[:, 3] = -1.0
                eng.nodes_gpu[:, 4] = 0.0
                eng.nodes_gpu[0, 3] = 0.0
                eng.nodes_gpu[0, 4] = 1.0
            else:
                eng.nodes_gpu[:, 3] = -1.0
                eng.nodes_gpu[:, 4] = 0.0


def finish_operation_and_evolve(eng: RealisticBattleEngine) -> None:
    """Vyhodnotí ukončenou operaci, zapíše telemetrii epochy, spustí evoluci a přepne scénář."""
    with eng._engine_lock:
        alive = eng.hp > 0.0
        if eng.epoch_winner < 0:
            scores = []
            for f in range(eng.n_factions):
                alive_c = int(((eng.teams == f) & alive).sum().item())
                kills_c = int(eng.kills_per_faction[f].item())
                pts_c = float(eng.node_control_points[f].item())
                scores.append(kills_c * 3.0 + pts_c * 2.5 + alive_c)
            eng.epoch_winner = int(np.argmax(scores))

        if eng.epoch_winner >= 0:
            eng.epoch_wins_per_faction[eng.epoch_winner] += 1
            eng.brains.generals[eng.epoch_winner].total_campaign_wins += 1

        # Bonus za přežití
        eng.fitness = torch.where(alive, eng.fitness + 18.0, eng.fitness)

        # Udělení XP generálům
        for f in range(eng.n_factions):
            f_mask = eng.teams == f
            f_kills = eng.kills_per_faction[f].item()
            f_nodes = eng.node_control_points[f].item()
            f_surv = (f_mask & alive).sum().item()
            earned_xp = f_kills * 4.0 + f_nodes * 2.5 + f_surv * 1.0
            if f == eng.epoch_winner:
                earned_xp += 40.0
            eng.brains.generals[f].xp += earned_xp

        # Výpočet globálního ROI reportu podle tříd jednotek
        class_roi_report = []
        for c in range(4):
            c_mask = eng.classes == c
            cnt = int(c_mask.sum().item())
            if cnt > 0:
                avg_fit = float(torch.mean(eng.fitness[c_mask]).item())
                tot_dmg = float(torch.sum(eng.damage_dealt[c_mask]).item())
                tot_dmg_taken = float(torch.sum(eng.damage_taken[c_mask]).item())
                tot_heal = float(torch.sum(eng.healing_done[c_mask]).item())
                tot_kills = int(torch.sum(eng.unit_kills[c_mask]).item())
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

        if eng.telemetry_mgr is not None:
            factions_stats = []
            for f in range(eng.n_factions):
                f_mask = eng.teams == f
                f_alive = (f_mask & alive).sum().item()
                f_fit = eng.fitness[f_mask]
                gen = eng.brains.generals[f]
                factions_stats.append({
                    "faction_id": f,
                    "level": gen.level,
                    "xp": float(gen.xp),
                    "alive_count": int(f_alive),
                    "kills": int(eng.kills_per_faction[f].item()),
                    "node_points": float(eng.node_control_points[f].item()),
                    "fitness_max": float(torch.max(f_fit).item()) if f_fit.numel() > 0 else 0.0,
                    "fitness_avg": float(torch.mean(f_fit).item()) if f_fit.numel() > 0 else 0.0,
                    "class_probs": [round(float(p), 3) for p in gen.get_class_probabilities()],
                })

            eng.telemetry_mgr.log_epoch(
                epoch_num=eng.current_epoch,
                scenario_name=eng.scenario.name,
                duration_ticks=int(eng.operation_time_sec * 60),
                winner_team=eng.epoch_winner,
                end_reason=eng.epoch_end_reason or "OPERAČNÍ VÍTĚZSTVÍ",
                factions_stats=factions_stats,
                total_wins=list(eng.epoch_wins_per_faction),
                class_roi=class_roi_report,
            )

        # Třídně specializovaná evoluce mozků
        eng.brains.evolve_epoch(
            brain_slots=eng.brain_slots,
            teams=eng.teams,
            classes=eng.classes,
            fitness=eng.fitness,
            n_factions=eng.n_factions,
            winner_team=eng.epoch_winner,
        )
        eng.current_epoch = eng.brains.epoch_number

        next_scenario = (eng.current_scenario_idx + 1) % 3
        eng.switch_scenario(next_scenario)


def compute_general_orders(
    eng: RealisticBattleEngine,
    dir_enemy: torch.Tensor,
    dist_enemy: torch.Tensor,
) -> torch.Tensor:
    """Vektorové rozkazy generála podle doktríny a stavu uzlů (plně vektorizováno bez Python cyklů)."""
    with eng._engine_lock:
        N = eng.N
        if eng.nodes_gpu.shape[0] == 0:
            return dir_enemy.clone()

        nodes_xy = eng.nodes_gpu[:, 0:2]
        node_owners = eng.nodes_gpu[:, 3]
        node_d = torch.cdist(eng.pos, nodes_xy)  # (N, num_nodes)
        num_nodes = node_d.shape[1]

        teams_long = eng.teams.long()
        owners_exp = node_owners.unsqueeze(0).expand(N, num_nodes)
        teams_exp = teams_long.unsqueeze(1).expand(N, num_nodes)
        is_owned = owners_exp == teams_exp
        is_unowned = ~is_owned

        # Doktríny frakcí
        doctrines = eng.faction_doctrine_tensor[teams_long]
        bias = is_unowned.float() * 140.0
        rush_bias = is_unowned.float() * 220.0
        rush_bias[:, 0] += 120.0
        hold_bias = is_owned.float() * 250.0
        skirmish_bias = torch.zeros_like(bias)
        if num_nodes > 1:
            skirmish_bias[:, 0] -= 180.0

        doc_bias = torch.where(
            doctrines.unsqueeze(1) == DOCTRINE_RUSH,
            rush_bias,
            torch.where(
                doctrines.unsqueeze(1) == DOCTRINE_HOLD,
                hold_bias,
                torch.where(doctrines.unsqueeze(1) == DOCTRINE_SKIRMISH, skirmish_bias, bias),
            ),
        )

        eff_d = node_d - doc_bias
        target_n = torch.argmin(eff_d, dim=1)
        vec = nodes_xy[target_n] - eng.pos
        norm = torch.norm(vec, dim=1, keepdim=True).clamp(min=1.0)
        return vec / norm
