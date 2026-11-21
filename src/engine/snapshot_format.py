"""
engine/snapshot_format.py - 2-fázový Stage -> Format pipeline pro telemetrii a HUD snapshoty.

Fáze 1 (Stage): extract_raw_telemetry_state() pod _engine_lock (< 0.3 ms), jeden sdružený GPU->CPU přenos.
Fáze 2 (Format): format_battle_snapshot() a format_inspected_unit() čistě na CPU mimo zámek enginu.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from ai_brains import CLASS_COSTS, UNLOCK_XP
from config import CLASS_NAMES

if TYPE_CHECKING:
    from engine_core import RealisticBattleEngine

STATE_NAMES = ("IDLE", "ENGAGE", "COVER_RETREAT", "SNIPER_LOCK", "TRIAGE_HEAL", "FORTRESS_PUSH")


def extract_raw_telemetry_state(
    eng: RealisticBattleEngine,
    target_id: int | None = None,
    auto_best: bool = True,
    include_inspection: bool = True,
) -> dict:
    """
    Fáze 1 (Stage): Minimální kritická sekce pod _engine_lock (< 0.3 ms).
    Stáhne sdružený balík GPU tenzorů do CPU paměti a oddělí měnitelné Python kolekce.
    """
    with eng._engine_lock:
        alive = eng.hp > 0.0
        N = eng.N

        # Sdružený balík 21 sloupců: 1 jediný PCIe přenos pro všechny per-unit atributy
        packed_gpu = torch.empty((N, 21), dtype=torch.float32, device=eng.device)
        packed_gpu[:, 0] = eng.pos[:, 0]
        packed_gpu[:, 1] = eng.pos[:, 1]
        packed_gpu[:, 2] = eng.heading
        packed_gpu[:, 3] = eng.hp
        packed_gpu[:, 4] = eng.class_max_hp
        packed_gpu[:, 5] = eng.teams.float()
        packed_gpu[:, 6] = eng.classes.float()
        packed_gpu[:, 7] = alive.float()
        packed_gpu[:, 8] = eng.suppression
        packed_gpu[:, 9] = eng.in_cover.float()
        packed_gpu[:, 10] = eng.reload_timer
        packed_gpu[:, 11] = eng.has_kinetic_shield.float() if hasattr(eng, "has_kinetic_shield") else 0.0
        packed_gpu[:, 12] = eng.unit_state.float() if hasattr(eng, "unit_state") else 0.0
        packed_gpu[:, 13] = eng.damage_dealt
        packed_gpu[:, 14] = eng.damage_taken
        packed_gpu[:, 15] = eng.unit_kills.float()
        packed_gpu[:, 16] = eng.fitness
        packed_gpu[:, 17] = eng.healing_done
        packed_gpu[:, 18] = eng.ammo_in_mag
        packed_gpu[:, 19] = eng.class_mag_size
        packed_gpu[:, 20] = eng.class_range

        packed_cpu = packed_gpu.detach().cpu().numpy()

        # Efekty a mapové tenzory (jednorázové stažení na CPU)
        tracers_p1_np = eng.active_tracers_p1.detach().cpu().numpy()
        tracers_p2_np = eng.active_tracers_p2.detach().cpu().numpy()
        tracers_team_np = eng.active_tracers_team.detach().cpu().numpy()
        heals_p1_np = eng.active_heals_p1.detach().cpu().numpy()
        heals_p2_np = eng.active_heals_p2.detach().cpu().numpy()
        sparks_np = eng.hit_sparks.detach().cpu().numpy()
        armor_sparks_np = eng.armor_sparks.detach().cpu().numpy()
        sniper_lasers_np = eng.sniper_lasers.detach().cpu().numpy()
        smokes_np = eng.active_smokes_gpu.detach().cpu().numpy()
        nodes_np = eng.nodes_gpu.detach().cpu().numpy()
        node_pts_np = eng.node_control_points.detach().cpu().numpy()
        kills_per_faction_np = eng.kills_per_faction.detach().cpu().numpy()
        gate_hp_np = (
            eng.gate_hp.detach().cpu().numpy()
            if eng.gate_hp.numel() > 0
            else np.empty((0,), dtype=np.float32)
        )

        # Kopie metadat generálů a bran (mimo mutaci při běhu další epochy)
        generals_raw = []
        for f in range(eng.n_factions):
            gen = eng.brains.generals[f]
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
            for gi, g in enumerate(getattr(eng.scenario, "gates", []))
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
                    b_slot = int(eng.brain_slots[resolved_id].item())
                    flow = eng.brains.inspect_decision_flow(b_slot, eng.last_sensors[resolved_id])
                    tar_idx = int(eng.last_target_enemy[resolved_id].item()) if hasattr(eng, "last_target_enemy") else -1
                    has_los = bool(eng.has_los_target[resolved_id].item()) if hasattr(eng, "has_los_target") else False
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
            "n_factions": int(eng.n_factions),
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
            "initial_team_counts": list(eng.initial_team_counts),
            "particles": [dict(p) for p in eng.particles],
            "walls": [list(w) for w in eng.current_walls],
            "pillars": list(eng.scenario.pillars),
            "bases": list(eng.scenario.bases),
            "events": list(eng.combat_events),
            "scenario_name": eng.scenario.name,
            "scenario_idx": int(eng.current_scenario_idx),
            "current_epoch": int(eng.current_epoch),
            "operation_id": int(getattr(eng, "operation_id", 0)),
            "operation_time_sec": float(eng.operation_time_sec),
            "epoch_wins_per_faction": list(eng.epoch_wins_per_faction),
            "faction_doctrines": list(eng.faction_doctrines),
            "faction_techs": [list(t) for t in eng.faction_active_techs],
            "history_avg_fitness": [list(h) for h in eng.brains.history_avg_fitness],
            "history_best_fitness": [list(h) for h in eng.brains.history_best_fitness],
            "inspected_raw": inspected_raw,
        }


def format_battle_snapshot(raw_state: dict) -> dict:
    """
    Fáze 2 (Format): Čistě CPU sestavení snapshotu pro HUD mimo zámek enginu.
    Nulový přístup k enginu i GPU.
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
