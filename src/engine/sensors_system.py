"""
engine/sensors_system.py - Detekce cílů, Line-of-Sight, léčení medika a egocentrické senzory.

Bezstavový subsystém pro Fázi 1 simulačního ticku:
- Detekce nejbližších nepřátel (amortizovaný 3-tick LoS raycast + rychlý cache update)
- Vyhledávání zraněných spojenců a léčení Medika (Zero-Trust zápis do staging_buf)
- Egocentrická transformace 14 senzorů do soustavy těla vojáka
- Forward pass neuronové sítě (brains.forward)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ai_brains import CLASS_MEDIC
from engine.spatial_system import check_line_of_sight

if TYPE_CHECKING:
    from engine.tick_context import TickContext
    from engine_core import RealisticBattleEngine


def run_sensors_and_brain(eng: RealisticBattleEngine, ctx: TickContext) -> None:
    """Vyhodnotí senzory, detekci nepřátel, léčení medika a provede forward pass mozku."""
    dt = ctx.dt
    N = ctx.N
    alive = ctx.alive
    alive_idx = ctx.alive_idx
    K = ctx.K

    # Detekce nepřátel a spojenců (předalokované buffery s automatickou změnou velikosti při změně N)
    if not hasattr(eng, "_buf_dir_enemy") or eng._buf_dir_enemy.shape[0] != N:
        eng._buf_dir_enemy = torch.zeros((N, 2), dtype=torch.float32, device=eng.device)
        eng._buf_dist_enemy = torch.full((N,), 9999.0, dtype=torch.float32, device=eng.device)
        eng._buf_target_enemy = torch.full((N,), -1, dtype=torch.long, device=eng.device)
        eng._buf_medic_patient_dir = torch.zeros((N, 2), dtype=torch.float32, device=eng.device)
        eng._buf_medic_has_patient = torch.zeros((N,), dtype=torch.bool, device=eng.device)
        eng._empty_xy = torch.zeros((0, 2), dtype=torch.float32, device=eng.device)
        eng._empty_i8 = torch.zeros((0,), dtype=torch.int8, device=eng.device)
    else:
        eng._buf_dir_enemy.zero_()
        eng._buf_dist_enemy.fill_(9999.0)
        eng._buf_target_enemy.fill_(-1)
        eng._buf_medic_patient_dir.zero_()
        eng._buf_medic_has_patient.zero_()

    dir_enemy = eng._buf_dir_enemy
    dist_enemy = eng._buf_dist_enemy
    target_enemy = eng._buf_target_enemy
    medic_patient_dir = eng._buf_medic_patient_dir
    medic_has_patient = eng._buf_medic_has_patient

    eng.active_heals_p1 = eng._empty_xy
    eng.active_heals_p2 = eng._empty_xy

    p_alive = None
    t_alive = None
    dist_mat = None
    diff_all = None
    same_team = None

    if K > 1:
        p_alive = eng.pos[alive_idx]
        t_alive = eng.teams[alive_idx]
        dist_mat = torch.cdist(p_alive, p_alive)
        diff_all = p_alive.unsqueeze(0) - p_alive.unsqueeze(1)
        same_team = t_alive.unsqueeze(1) == t_alive.unsqueeze(0)

        # Analytické ověření Line-of-Sight (Zákaz Wallhacku / koukání skrze zdi a překážky):
        # Amortizace: plný O(K^2) výpočet FOV + raycast provádíme pouze každý 3. tick!
        need_full_los = (eng.tick_counter % 3 == 0) or not hasattr(eng, "_cached_target_enemy")
        if need_full_los:
            head_alive = eng.heading[alive_idx]
            angle_to_tar = torch.atan2(diff_all[:, :, 1], diff_all[:, :, 0])
            angle_diff = (angle_to_tar - head_alive.unsqueeze(1) + math.pi) % (2.0 * math.pi) - math.pi
            in_fov = torch.abs(angle_diff) < (math.radians(140) * 0.5)

            c_alive = eng.classes[alive_idx]
            max_vis = eng.CONST_CLASS_MAX_VIS[c_alive.long()].unsqueeze(1)

            # TECH_RADAR: +25% dosah viditelnosti pro frakce s odemčeným radarem
            for f in range(eng.n_factions):
                if 1 in eng.faction_active_techs[f]:  # TECH_RADAR = 1
                    f_in_alive = t_alive == f
                    if f_in_alive.any():
                        max_vis[f_in_alive] *= 1.25

            # Proximitní slyšení kroků za zády povoleno pouze na kontaktní vzdálenost do 75 px
            can_spot = (~same_team) & (dist_mat <= max_vis) & (in_fov | (dist_mat < 75.0))
            vis_dists = torch.where(can_spot, dist_mat, 9999.0)

            top_k = min(4, vis_dists.shape[1])
            vals, args = torch.topk(vis_dists, k=top_k, dim=1, largest=False)

            cand_flat = args.reshape(-1)
            p_from_flat = p_alive.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, 2)
            p_to_flat = p_alive[cand_flat]
            max_los_diag = float(math.hypot(eng.W, eng.H))
            valid_range = vals.reshape(-1) < max_los_diag

            los_flat = torch.zeros(K * top_k, dtype=torch.bool, device=eng.device)
            active_pairs = torch.where(valid_range)[0]
            if active_pairs.numel() > 0:
                los_flat[active_pairs] = check_line_of_sight(
                    p_from_flat[active_pairs], p_to_flat[active_pairs],
                    eng.pillars_gpu, eng.walls_gpu, eng.active_smokes_gpu
                )

            los_grid = los_flat.reshape(K, top_k)
            has_los = los_grid.any(dim=1)
            first_col = torch.argmax(los_grid.int(), dim=1)
            unit_arange = torch.arange(K, device=eng.device)
            best_t = args[unit_arange, first_col]
            best_dist = vals[unit_arange, first_col]

            # target_enemy má platné ID POUZE pokud má jednotka ověřenou přímou viditelnost (has_los == True)
            target_enemy[alive_idx] = torch.where(has_los, alive_idx[best_t], -1)
            dist_enemy[alive_idx] = torch.where(has_los, best_dist, 9999.0)

            e_vec = p_alive[best_t] - p_alive
            direct_dir = e_vec / best_dist.unsqueeze(1).clamp(min=1.0)
            dir_enemy[alive_idx] = torch.where(has_los.unsqueeze(1), direct_dir, 0.0)

            eng.has_los_target = torch.zeros((N,), dtype=torch.bool, device=eng.device)
            eng.has_los_target[alive_idx] = has_los

            eng._cached_target_enemy = target_enemy.clone()
            eng._cached_dist_enemy = dist_enemy.clone()
            eng._cached_dir_enemy = dir_enemy.clone()
        else:
            # Rychlé aktualizování vzdálenosti a směru k dříve zaměřeným cílům bez drahého raycastu
            target_enemy.copy_(eng._cached_target_enemy)
            has_cached_tar = (target_enemy >= 0) & alive & (target_enemy < N)
            if has_cached_tar.any():
                c_idx = torch.where(has_cached_tar)[0]
                tar_ids = target_enemy[c_idx]
                tar_alive = (eng.hp[tar_ids] > 0.0) & (eng.teams[tar_ids] != eng.teams[c_idx])
                valid_c = c_idx[tar_alive]
                valid_tar = tar_ids[tar_alive]
                if valid_c.numel() > 0:
                    v_diff = eng.pos[valid_tar] - eng.pos[valid_c]
                    v_dist = torch.norm(v_diff, dim=1)
                    v_dir = v_diff / v_dist.unsqueeze(1).clamp(min=1.0)
                    dist_enemy[valid_c] = v_dist
                    dir_enemy[valid_c] = v_dir
                invalid_c = c_idx[~tar_alive]
                if invalid_c.numel() > 0:
                    target_enemy[invalid_c] = -1
                    dist_enemy[invalid_c] = 9999.0
                    dir_enemy[invalid_c] = 0.0
            eng.has_los_target = (target_enemy >= 0) & alive

        # Léčení Medika
        c_alive = eng.classes[alive_idx]
        medics_rel = torch.where(c_alive == CLASS_MEDIC)[0]
        if medics_rel.numel() > 0:
            hp_alive = eng.hp[alive_idx]
            max_hp_alive = eng.class_max_hp[alive_idx]
            injured_alive = hp_alive < (max_hp_alive - 1.0)

            m_dists = dist_mat[medics_rel]
            m_same = same_team[medics_rel]
            seek_patients = m_same & injured_alive.unsqueeze(0) & (m_dists > 1.0) & (m_dists <= 450.0)
            if seek_patients.any():
                patient_d = torch.where(seek_patients, m_dists, torch.full_like(m_dists, 9999.0))
                min_p_d, min_p_rel = torch.min(patient_d, dim=1)

                found_injured = min_p_d < 450.0
                if found_injured.any():
                    m_seek_rel = medics_rel[found_injured]
                    p_seek_rel = min_p_rel[found_injured]
                    m_seek_glob = alive_idx[m_seek_rel]
                    vec_to_pat = p_alive[p_seek_rel] - p_alive[m_seek_rel]
                    norm_to_pat = torch.norm(vec_to_pat, dim=1, keepdim=True).clamp(min=1.0)
                    medic_patient_dir[m_seek_glob] = vec_to_pat / norm_to_pat
                    medic_has_patient[m_seek_glob] = True

                can_heal = min_p_d <= 260.0
                if can_heal.any():
                    c_m_rel = medics_rel[can_heal]
                    c_p_rel = min_p_rel[can_heal]
                    p_m_pos = p_alive[c_m_rel]
                    p_p_pos = p_alive[c_p_rel]
                    heal_los = check_line_of_sight(p_m_pos, p_p_pos, eng.pillars_gpu, eng.walls_gpu, eng.active_smokes_gpu)
                    if heal_los.any():
                        active_m_rel = c_m_rel[heal_los]
                        active_p_rel = c_p_rel[heal_los]
                        m_global = alive_idx[active_m_rel]
                        p_global = alive_idx[active_p_rel]

                        heal_amount = 42.0 * dt
                        missing_hp = (eng.class_max_hp[p_global] - eng.hp[p_global]).clamp(min=0.0)
                        actual_heal = torch.minimum(torch.full_like(missing_hp, heal_amount), missing_hp)

                        # Zero-Trust zápis léčení do staging bufferu
                        eng.staging_buf.delta_hp.index_add_(0, p_global, actual_heal)
                        eng.healing_done[m_global] += actual_heal
                        eng.fitness[m_global] += actual_heal * 4.0

                        h_lim = min(m_global.numel(), 32)
                        eng.active_heals_p1 = eng.pos[m_global[:h_lim]]
                        eng.active_heals_p2 = eng.pos[p_global[:h_lim]]

    dir_order = eng._compute_general_orders(dir_enemy, dist_enemy)
    dir_order = torch.where(medic_has_patient.unsqueeze(1), medic_patient_dir, dir_order)

    # ==============================================================
    # EGOCENTRICKÁ TRANSFORMACE SENZORŮ DO SOUSTAVY TĚLA VOJÁKA
    # ==============================================================
    cos_th = torch.cos(eng.heading)
    sin_th = torch.sin(eng.heading)

    enemy_fwd = dir_enemy[:, 0] * cos_th + dir_enemy[:, 1] * sin_th
    enemy_right = -dir_enemy[:, 0] * sin_th + dir_enemy[:, 1] * cos_th

    fire_fwd = eng.incoming_fire_dir[:, 0] * cos_th + eng.incoming_fire_dir[:, 1] * sin_th
    fire_right = -eng.incoming_fire_dir[:, 0] * sin_th + eng.incoming_fire_dir[:, 1] * cos_th

    order_fwd = dir_order[:, 0] * cos_th + dir_order[:, 1] * sin_th
    order_right = -dir_order[:, 0] * sin_th + dir_order[:, 1] * cos_th

    # Vektory k nejbližšímu krytu (stěna / bunkr) - amortizováno na každý 3. tick
    if (eng.tick_counter % 3 == 0) or not hasattr(eng, "_cached_cover_dir"):
        cover_dir = torch.zeros((N, 2), dtype=torch.float32, device=eng.device)
        if eng.walls_gpu.shape[0] > 0:
            wall_centers_x = (eng.walls_gpu[:, 0] + eng.walls_gpu[:, 2]) * 0.5
            wall_centers_y = (eng.walls_gpu[:, 1] + eng.walls_gpu[:, 3]) * 0.5
            wall_centers = torch.stack([wall_centers_x, wall_centers_y], dim=1)
            w_diff = wall_centers.unsqueeze(0) - eng.pos.unsqueeze(1)  # (N, Nw, 2)
            w_dists = torch.norm(w_diff, dim=2)
            nearest_w = torch.argmin(w_dists, dim=1)
            closest_vec = wall_centers[nearest_w] - eng.pos
            c_norm = torch.norm(closest_vec, dim=1, keepdim=True).clamp(min=1.0)
            cover_dir = closest_vec / c_norm
        eng._cached_cover_dir = cover_dir
    else:
        cover_dir = eng._cached_cover_dir

    cover_fwd = cover_dir[:, 0] * cos_th + cover_dir[:, 1] * sin_th
    cover_right = -cover_dir[:, 0] * sin_th + cover_dir[:, 1] * cos_th

    # Koheze čety a komunikační signál (hustota spojenců v okolí do 140px a jejich Squad Ping)
    cohesion_and_ping = torch.zeros((N,), dtype=torch.float32, device=eng.device)
    if K > 1:
        ally_nearby = same_team & (dist_mat < 140.0) & (dist_mat > 1.0)
        ally_cnt = ally_nearby.sum(dim=1).float()
        cohesion_base = (ally_cnt / 8.0).clamp(0.0, 1.0) * 2.0 - 1.0

        # Squad Ping spojenců v okruhu 140 px (vážený průměr signálů čety)
        ally_ping = eng.squad_ping[alive_idx].unsqueeze(0).expand(K, K)
        ping_masked = torch.where(ally_nearby, ally_ping, torch.zeros_like(ally_ping))
        ping_sum = ping_masked.sum(dim=1)
        avg_ally_ping = torch.where(ally_cnt > 0.0, ping_sum / ally_cnt, torch.zeros_like(ally_cnt))

        # Fúze koheze (0.7) a signálu čety (0.3) v rozsahu [-1.0, 1.0]
        cohesion_and_ping[alive_idx] = (cohesion_base * 0.70 + avg_ally_ping * 0.30).clamp(-1.0, 1.0)

    # Taktický stav zóny (-1 = nepřítel dominuje, 0 = neutrál/volno, 1 = plně pod kontrolou)
    zone_status = torch.zeros((N,), dtype=torch.float32, device=eng.device)
    if eng.nodes_gpu.shape[0] > 0 and K > 1:
        nodes_dists = torch.cdist(eng.pos, eng.nodes_gpu[:, 0:2])
        closest_n = torch.argmin(nodes_dists, dim=1)
        n_owner = eng.nodes_gpu[closest_n, 3]
        is_ours = n_owner == eng.teams.float()
        is_enemy = (n_owner >= 0) & (~is_ours)
        zone_status = torch.where(is_ours, 1.0, torch.where(is_enemy, -1.0, 0.0))

    if eng._sensor_buf is None or eng._sensor_buf.shape[0] != N:
        eng._sensor_buf = torch.zeros((N, 14), dtype=torch.float32, device=eng.device)
    sensors = eng._sensor_buf
    sensors.zero_()
    sensors[:, 0] = enemy_fwd
    sensors[:, 1] = enemy_right
    # Nelineární hyperbolické škálování vzdálenosti cíle s vysokou rozlišovací schopností nablízko
    sensors[:, 2] = (2.0 / (1.0 + dist_enemy / (eng.class_range * 0.35)) - 1.0).clamp(-1.0, 1.0)
    sensors[:, 3] = fire_fwd
    sensors[:, 4] = fire_right
    sensors[:, 5] = order_fwd
    sensors[:, 6] = order_right
    sensors[:, 7] = eng.suppression
    sensors[:, 8] = ((eng.ammo_in_mag / eng.class_mag_size) * 2.0 - 1.0).clamp(-1.0, 1.0)
    sensors[:, 9] = (eng.hp / eng.class_max_hp) * 2.0 - 1.0
    sensors[:, 10] = cover_fwd
    sensors[:, 11] = cover_right
    sensors[:, 12] = cohesion_and_ping
    sensors[:, 13] = zone_status

    eng.last_sensors = sensors
    brain_out = eng.brains.forward(eng.brain_slots, sensors)

    eng.incoming_fire_dir.mul_(0.85)

    # Uložení mezivýsledků do TickContextu (bez kopií, pouze reference na GPU tenzory)
    ctx.dir_enemy = dir_enemy
    ctx.dist_enemy = dist_enemy
    ctx.target_enemy = target_enemy
    ctx.medic_patient_dir = medic_patient_dir
    ctx.medic_has_patient = medic_has_patient
    ctx.p_alive = p_alive
    ctx.t_alive = t_alive
    ctx.dist_mat = dist_mat
    ctx.diff_all = diff_all
    ctx.same_team = same_team
    ctx.cos_th = cos_th
    ctx.sin_th = sin_th
    ctx.brain_out = brain_out
