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
