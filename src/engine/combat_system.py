"""
combat_system.py - Správa bojových mechanik, třídních schopností a pasivních efektů.

Podporuje 4 třídní schopnosti (řízené tenzorem mozku Action 5 - Ability):
  - Grunt: Taktická kouřová clona (podpora TECH_AEROSOL pro +40 % poloměr)
  - Medic: Bio-pulz (hromadné léčení v okolí) + pasivní triage léčení do StagingBufferu
  - Sniper: Focus zaměření (+45 % zorný dosah, snížení rozptylu)
  - Juggernaut: Fortress Shield (přechodné navýšení kinetické a balistické odolnosti)
"""

import math
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

        # 3. Sniper: Focus Zaměření (vektorizovaně pro všechny aktivující snipry)
        sniper_mask = wants_ability & (classes == CLASS_SNIPER)
        if sniper_mask.any():
            ability_cooldown = torch.where(sniper_mask, torch.full_like(ability_cooldown, 20.0), ability_cooldown)
            sniper_focus_timer = torch.where(sniper_mask, torch.full_like(sniper_focus_timer, 5.0), sniper_focus_timer)
            fitness = torch.where(sniper_mask, fitness + 2.0, fitness)

        # 4. Juggernaut: Fortress Shield (vektorizovaně pro všechny aktivující juggernauty)
        jugg_mask = wants_ability & (classes == CLASS_JUGGERNAUT)
        if jugg_mask.any():
            ability_cooldown = torch.where(jugg_mask, torch.full_like(ability_cooldown, 22.0), ability_cooldown)
            fortress_shield_timer = torch.where(jugg_mask, torch.full_like(fortress_shield_timer, 6.0), fortress_shield_timer)
            fitness = torch.where(jugg_mask, fitness + 2.5, fitness)

        # 2. Medic: Bio-Pulz
        medic_mask = wants_ability & (classes == CLASS_MEDIC)
        if medic_mask.any():
            ability_cooldown = torch.where(medic_mask, torch.full_like(ability_cooldown, 15.0), ability_cooldown)
            med_indices = torch.where(medic_mask)[0]
            for m_uid in med_indices:
                f_id = teams[m_uid]
                m_pos = pos[m_uid:m_uid+1]
                team_mask = alive & (teams == f_id)
                if team_mask.any():
                    t_indices = torch.where(team_mask)[0]
                    dists = torch.norm(pos[t_indices] - m_pos, dim=1)
                    in_radius = dists <= 180.0
                    if in_radius.any():
                        pulse_targets = t_indices[in_radius]
                        pulse_amount = 35.0
                        missing_hp = (class_max_hp[pulse_targets] - hp[pulse_targets]).clamp(min=0.0)
                        actual_pulse = torch.minimum(torch.full_like(missing_hp, pulse_amount), missing_hp)

                        staging_buf.delta_hp.index_add_(0, pulse_targets, actual_pulse)
                        total_healed = actual_pulse.sum()
                        healing_done[m_uid] += total_healed
                        fitness[m_uid] += total_healed * 3.5

        # 1. Grunt: Taktický kouř
        grunt_mask = wants_ability & (classes == CLASS_GRUNT)
        if grunt_mask.any():
            grunt_indices = torch.where(grunt_mask)[0]
            new_smokes = []
            for g_uid in grunt_indices:
                uid = int(g_uid.item())
                f_id = int(teams[uid].item())
                team_smokes = 0
                if active_smokes_gpu.shape[0] > 0:
                    team_smokes = int((active_smokes_gpu[:, 4] == float(f_id)).sum().item())
                if team_smokes < 3:
                    radius = 55.0
                    duration = 6.0
                    if faction_techs and TECH_AEROSOL in faction_techs[f_id]:
                        radius *= 1.40
                        duration *= 1.25

                    ux, uy = float(pos[uid, 0].item()), float(pos[uid, 1].item())
                    new_smokes.append([ux, uy, radius, duration, float(f_id)])
                    ability_cooldown[uid] = 18.0

                    if visual_mode and particles is not None:
                        for _ in range(8):
                            ang = float(torch.rand(1).item() * 2 * math.pi)
                            spd = float(torch.rand(1).item() * 2.0 + 0.5)
                            particles.append({
                                "x": ux, "y": uy,
                                "vx": math.cos(ang) * spd,
                                "vy": math.sin(ang) * spd,
                                "life": 1.2,
                                "col": (120, 140, 130),
                            })
            if new_smokes:
                new_t = torch.tensor(new_smokes, dtype=torch.float32, device=self.device)
                active_smokes_gpu = torch.cat([active_smokes_gpu, new_t], dim=0)

        return ability_cooldown, active_smokes_gpu, fortress_shield_timer, sniper_focus_timer
