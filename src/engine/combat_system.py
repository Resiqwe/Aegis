"""
combat_system.py - Správa bojových mechanik, třídních schopností a pasivních efektů.

Podporuje 4 třídní schopnosti (řízené tenzorem mozku Action 5 - Ability):
  - Grunt: Taktická kouřová clona (podpora TECH_AEROSOL pro +40 % poloměr)
  - Medic: Bio-pulz (hromadné léčení v okolí) + pasivní triage léčení do StagingBufferu
  - Sniper: Focus zaměření (+45 % zorný dosah, snížení rozptylu)
  - Juggernaut: Fortress Shield (přechodné navýšení kinetické a balistické odolnosti)
"""

import math
import numpy as np
import torch
from config import (
    CLASS_GRUNT, CLASS_MEDIC, CLASS_SNIPER, CLASS_JUGGERNAUT,
    TECH_AEROSOL
)
from engine.staging_buffer import SimulationStagingBuffer


class CombatSystem:
    def __init__(self, device: torch.device):
        self.device = device

    def process_abilities(
        self,
        dt: float,
        brain_ability: torch.Tensor,
        alive: torch.Tensor,
        classes: torch.Tensor,
        teams: torch.Tensor,
        pos: torch.Tensor,
        hp: torch.Tensor,
        class_max_hp: torch.Tensor,
        ability_cooldown: torch.Tensor,
        active_smokes_gpu: torch.Tensor,
        fortress_shield_timer: torch.Tensor,
        sniper_focus_timer: torch.Tensor,
        staging_buf: SimulationStagingBuffer,
        healing_done: torch.Tensor,
        fitness: torch.Tensor,
        active_heals_p1: list,
        active_heals_p2: list,
        faction_techs: list[set[int]] | None = None,
        visual_mode: bool = False,
        particles: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vyhodnotí aktivní schopnosti vojáků podle výstupu sítě (brain_ability > 0.60).
        Vrací (ability_cooldown, active_smokes_gpu, fortress_shield_timer, sniper_focus_timer).
        """
        # Odpočet cooldownů a trvání buffů
        ability_cooldown = (ability_cooldown - dt).clamp(min=0.0)
        fortress_shield_timer = (fortress_shield_timer - dt).clamp(min=0.0)
        sniper_focus_timer = (sniper_focus_timer - dt).clamp(min=0.0)

        # Aktualizace aktivních kouřů [x, y, radius, timer, team]
        if active_smokes_gpu.shape[0] > 0:
            active_smokes_gpu[:, 3] -= dt
            keep_smoke = active_smokes_gpu[:, 3] > 0.0
            active_smokes_gpu = active_smokes_gpu[keep_smoke]

        wants_ability = alive & (brain_ability > 0.60) & (ability_cooldown <= 0.0)
        if not wants_ability.any():
            return ability_cooldown, active_smokes_gpu, fortress_shield_timer, sniper_focus_timer

        # 3. Sniper: Focus Zaměření (vektorizovaně bez alokace full_like)
        sniper_mask = wants_ability & (classes == CLASS_SNIPER)
        if sniper_mask.any():
            ability_cooldown = torch.where(sniper_mask, 20.0, ability_cooldown)
            sniper_focus_timer = torch.where(sniper_mask, 5.0, sniper_focus_timer)
            fitness = torch.where(sniper_mask, fitness + 2.0, fitness)

        # 4. Juggernaut: Fortress Shield (vektorizovaně bez alokace full_like)
        jugg_mask = wants_ability & (classes == CLASS_JUGGERNAUT)
        if jugg_mask.any():
            ability_cooldown = torch.where(jugg_mask, 22.0, ability_cooldown)
            fortress_shield_timer = torch.where(jugg_mask, 6.0, fortress_shield_timer)
            fitness = torch.where(jugg_mask, fitness + 2.5, fitness)

        # 2. Medic: Bio-Pulz (plně vektorizovaně přes matici (M, N) bez Python smyčky)
        medic_mask = wants_ability & (classes == CLASS_MEDIC)
        if medic_mask.any():
            ability_cooldown = torch.where(medic_mask, 15.0, ability_cooldown)
            med_indices = torch.where(medic_mask)[0]
            dists = torch.cdist(pos[med_indices], pos)  # (M, N)
            in_range = (
                (dists <= 180.0)
                & alive.unsqueeze(0)
                & (teams.unsqueeze(0) == teams[med_indices].unsqueeze(1))
            )
            pulse_each = (class_max_hp - hp).clamp(0.0, 35.0)  # (N,)
            healed_mat = torch.where(in_range, pulse_each.unsqueeze(0), 0.0)  # (M, N)
            staging_buf.delta_hp += healed_mat.sum(dim=0)
            total_healed = healed_mat.sum(dim=1)
            healing_done[med_indices] += total_healed
            fitness[med_indices] += total_healed * 3.5

        # 1. Grunt: Taktický kouř (dávkový přenos bez .item() uvnitř smyčky)
        grunt_mask = wants_ability & (classes == CLASS_GRUNT)
        if grunt_mask.any():
            grunt_indices = torch.where(grunt_mask)[0]
            g_ids_cpu = grunt_indices.detach().cpu().numpy()
            g_teams_cpu = teams[grunt_indices].detach().cpu().numpy()
            g_pos_cpu = pos[grunt_indices].detach().cpu().numpy()

            team_smoke_counts = [0, 0, 0, 0]
            if active_smokes_gpu.shape[0] > 0:
                smoke_teams_cpu = active_smokes_gpu[:, 4].detach().cpu().numpy()
                for st in smoke_teams_cpu:
                    ti = int(st)
                    if 0 <= ti < 4:
                        team_smoke_counts[ti] += 1

            new_smokes = []
            activated_uids = []
            for idx_g in range(len(g_ids_cpu)):
                uid = int(g_ids_cpu[idx_g])
                f_id = int(g_teams_cpu[idx_g])
                if 0 <= f_id < 4 and team_smoke_counts[f_id] < 3:
                    radius = 55.0
                    duration = 6.0
                    if faction_techs and TECH_AEROSOL in faction_techs[f_id]:
                        radius *= 1.40
                        duration *= 1.25

                    ux = float(g_pos_cpu[idx_g, 0])
                    uy = float(g_pos_cpu[idx_g, 1])
                    new_smokes.append([ux, uy, radius, duration, float(f_id)])
                    activated_uids.append(uid)

                    if visual_mode and particles is not None:
                        for _ in range(8):
                            ang = float(np.random.rand() * 2.0 * math.pi)
                            spd = float(np.random.rand() * 2.0 + 0.5)
                            particles.append({
                                "x": ux, "y": uy,
                                "vx": math.cos(ang) * spd,
                                "vy": math.sin(ang) * spd,
                                "life": 1.2,
                                "col": (120, 140, 130),
                            })
            if activated_uids:
                act_t = torch.tensor(activated_uids, dtype=torch.long, device=self.device)
                ability_cooldown[act_t] = 18.0
            if new_smokes:
                new_t = torch.tensor(new_smokes, dtype=torch.float32, device=self.device)
                active_smokes_gpu = torch.cat([active_smokes_gpu, new_t], dim=0)

        return ability_cooldown, active_smokes_gpu, fortress_shield_timer, sniper_focus_timer


def run_weapons_and_firing(eng, ctx) -> None:
    """
    Fáze zbraní a střelby před atomickým commitem (_atomic_commit_pipeline):
    - Deterministický automat přebíjení
    - Vyhodnocení spouště (Trigger > 0.58) a LoS ověření střelby
    - Vektorová balistika (VectorizedBallisticsSystem) a Zero-Trust zápis delt do staging_buf
    """
    from engine.spatial_system import check_line_of_sight

    DOCTRINE_HOLD = 2
    alive = ctx.alive
    alive_idx = ctx.alive_idx
    K = ctx.K
    dist_enemy = ctx.dist_enemy
    target_enemy = ctx.target_enemy
    p_alive = ctx.p_alive
    brain_out = ctx.brain_out
    has_combat_target = ctx.has_target
    tar_diff = ctx.tar_diff

    # DETERMINISTICKÝ AUTOMAT PŘEBÍJENÍ (Bez zbytečného zatěžování mozku přebíjením):
    empty_mag = eng.ammo_in_mag <= 0
    tactical_cover_reload = eng.in_cover & (eng.ammo_in_mag < eng.class_mag_size * 0.45) & (~has_combat_target)
    start_reload = (
        alive
        & (empty_mag | tactical_cover_reload)
        & (eng.reload_timer <= 0.0)
        & (eng.ammo_in_mag < eng.class_mag_size)
    )
    eng.reload_timer = torch.where(start_reload, eng.CONST_RELOAD_TIME, eng.reload_timer)

    # Střelba: Výstup 3 je Trigger (Sigmoid [0, 1] > 0.58)
    eng.cooldown = (eng.cooldown - 1).clamp(min=0)

    valid_tar = (target_enemy >= 0) & (target_enemy < eng.N)
    # tar_diff už máme spočítaný ze stejných pozic a headingů výše -> 0 duplicitních atan2/nonzero!
    on_target = alive & valid_tar & (torch.abs(tar_diff) < math.radians(45))

    # Trigger je nyní čistý Sigmoid práh > 0.58 z výstupu 3
    brain_trigger = brain_out[:, 3] > 0.58

    can_fire = (
        alive
        & valid_tar
        & eng.has_los_target
        & on_target
        & brain_trigger
        & (eng.ammo_in_mag > 0)
        & (eng.reload_timer <= 0.0)
        & (eng.cooldown == 0)
        & (dist_enemy <= eng.class_range)
    )
    shooters = torch.where(can_fire)[0]

    eng.active_tracers_p1 = eng._empty_xy
    eng.active_tracers_p2 = eng._empty_xy
    eng.active_tracers_team = eng._empty_i8
    eng.hit_sparks = eng._empty_xy
    eng.armor_sparks = eng._empty_xy

    if shooters.numel() > 0:
        targets = target_enemy[shooters]
        p_from = eng.pos[shooters]
        p_to = eng.pos[targets]

        los_ok = check_line_of_sight(p_from, p_to, eng.pillars_gpu, eng.walls_gpu, eng.active_smokes_gpu)
        valid_s = shooters[los_ok]
        valid_t = targets[los_ok]

        if valid_s.numel() > 0:
            eng.ammo_in_mag[valid_s] -= 1.0
            s_classes = eng.classes[valid_s].long()
            cd_val = eng.CONST_CLASS_COOLDOWNS[s_classes]
            eng.cooldown[valid_s] = cd_val

            fire_vec = eng.pos[valid_s] - eng.pos[valid_t]
            fire_norm = torch.norm(fire_vec, dim=1, keepdim=True).clamp(min=1.0)
            eng.incoming_fire_dir[valid_t] = fire_vec / fire_norm

            # Přesnost podle třídy
            base_acc = eng.CONST_CLASS_BASE_ACC[s_classes]
            cover_mult = torch.where(s_classes == CLASS_SNIPER, 0.60, 0.48)
            t_doc = eng.faction_doctrine_tensor[eng.teams[valid_t].long()]
            hold_bonus = torch.where(t_doc == DOCTRINE_HOLD, 0.85, 1.0)

            hit_prob = torch.where(eng.in_cover[valid_t], base_acc * cover_mult * hold_bonus, base_acc)
            # Sniper Focus buff: +0.20 přesnost pro snipry s aktivním fokusem
            sniper_focus_active = eng.sniper_focus_timer[valid_s] > 0.0
            hit_prob = torch.where(sniper_focus_active, (hit_prob + 0.20).clamp(max=0.98), hit_prob)
            hit_prob = hit_prob * (1.0 - eng.suppression[valid_s] * 0.45)

            is_hit = torch.rand(valid_s.numel(), device=eng.device) < hit_prob

            # ==============================================================
            # AEGIS 2.0 VEKTOROVÁ BALISTIKA: Úhlový pancíř, Zónový rozpad & Kinetický Ragdoll
            # ==============================================================
            b_res = eng.ballistics.resolve_ballistic_hits(
                valid_s=valid_s,
                valid_t=valid_t,
                is_hit=is_hit,
                shooter_pos=eng.pos[valid_s],
                target_pos=eng.pos[valid_t],
                target_heading=eng.heading[valid_t],
                shooter_classes=eng.classes[valid_s],
                target_classes=eng.classes[valid_t],
                target_in_cover=eng.in_cover[valid_t],
                has_kinetic_shield=(
                    eng.has_kinetic_shield
                    if hasattr(eng, "has_kinetic_shield")
                    else torch.zeros_like(eng.hp, dtype=torch.bool)
                ),
                base_dmg=eng.class_dmg[valid_s],
                sniper_focus_active=eng.sniper_focus_timer[valid_s] > 0.0,
                fortress_shield_active=eng.fortress_shield_timer[valid_t] > 0.0,
            )

            dmg = b_res["final_damage"]
            # Prasknutí kinetických štítů při absorpci
            if hasattr(eng, "has_kinetic_shield") and b_res["shield_absorbed"].any():
                eng.has_kinetic_shield[valid_t[b_res["shield_absorbed"]]] = False

            # Vizuální jiskry odraženého pancíře (Armor Sparks)
            if eng.visual_mode and b_res["flank_hit"].any():
                eng.armor_sparks = eng.pos[valid_t[b_res["flank_hit"]][:24]]

            # ==============================================================
            # ZERO-TRUST WRITE-ONLY BUFFER: Všechny impulzy a delty se ukládají zde!
            # ==============================================================
            # Kinetický impuls (Knockback)
            kb_vectors = torch.stack([b_res["knockback_x"], b_res["knockback_y"]], dim=1)
            eng.staging_buf.delta_vel.index_add_(0, valid_t, kb_vectors)

            # Poškození (záporné HP) a statistiky zásahu
            eng.staging_buf.delta_hp.index_add_(0, valid_t, -dmg)
            eng.staging_buf.damage_taken.index_add_(0, valid_t, dmg)

            # Záznam střelce pro přesnou atribuci killů
            if is_hit.any():
                hit_t = valid_t[is_hit]
                hit_s = valid_s[is_hit]
                eng.staging_buf.last_attacker_id[hit_t] = hit_s

            # Suppression yield
            eng.staging_buf.delta_suppression.index_add_(0, valid_t, b_res["suppression_yield"])

            # Motoric Stagger duration (zapíše se nejvyšší hodnota trvání)
            if b_res["stagger_mask"].any():
                stagg_idx = valid_t[b_res["stagger_mask"]]
                stagg_dur = b_res["stagger_duration"][b_res["stagger_mask"]]
                eng.staging_buf.stagger_duration[stagg_idx] = torch.maximum(
                    eng.staging_buf.stagger_duration[stagg_idx], stagg_dur
                )

            # Pud sebezáchovy: penalizace za poškození utržené mimo kryt
            out_of_cover_hit = is_hit & (~eng.in_cover[valid_t])
            if out_of_cover_hit.any():
                pen_targets = valid_t[out_of_cover_hit]
                pen_dmg = (dmg * 0.22)[out_of_cover_hit]
                eng.staging_buf.out_of_cover_penalty.index_add_(0, pen_targets, pen_dmg)

            # Near-miss suppression pro jednotky v okolí zásahu
            if K > 1 and valid_t.numel() > 0:
                d_to_targets = torch.cdist(p_alive, eng.pos[valid_t])
                near_miss = (d_to_targets < 65.0).any(dim=1)
                if near_miss.any():
                    supp_targets = alive_idx[near_miss]
                    supp_vals = torch.full((supp_targets.numel(),), 0.10, device=eng.device)
                    eng.staging_buf.delta_suppression.index_add_(0, supp_targets, supp_vals)

            eng.damage_dealt[valid_s] += dmg
            eng.fitness[valid_s] += dmg * 0.65

            if eng.visual_mode:
                s_lim = min(valid_s.numel(), 64)
                eng.active_tracers_p1 = eng.pos[valid_s[:s_lim]]
                eng.active_tracers_p2 = eng.pos[valid_t[:s_lim]]
                eng.active_tracers_team = eng.teams[valid_s[:s_lim]]
                if is_hit.any():
                    eng.hit_sparks = eng.pos[valid_t[is_hit][:32]]
                    is_target_jugg = eng.classes[valid_t].long() == CLASS_JUGGERNAUT
                    jugg_hits = is_hit & is_target_jugg
                    if jugg_hits.any():
                        eng.armor_sparks = eng.pos[valid_t[jugg_hits][:24]]
