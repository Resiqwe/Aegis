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

        trigger_indices = torch.nonzero(wants_ability).squeeze(1)
        new_smokes = []

        for idx in trigger_indices:
            uid = int(idx.item())
            c_id = int(classes[uid].item())
            f_id = int(teams[uid].item())

            # 1. Grunt: Taktický kouř
            if c_id == CLASS_GRUNT:
                team_smokes = 0
                if active_smokes_gpu.shape[0] > 0:
                    team_smokes = int((active_smokes_gpu[:, 4] == f_id).sum().item())
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

            # 2. Medic: Bio-Pulz (hromadné nárazové vyléčení všech spojenců v rádiusu 180 px)
            elif c_id == CLASS_MEDIC:
                ability_cooldown[uid] = 15.0
                m_pos = pos[uid]
                team_mask = alive & (teams == f_id)
                t_indices = torch.nonzero(team_mask).squeeze(1)
                dists = torch.norm(pos[t_indices] - m_pos, dim=1)
                in_radius = dists <= 180.0
                if in_radius.any():
                    pulse_targets = t_indices[in_radius]
                    pulse_amount = 35.0
                    missing_hp = (class_max_hp[pulse_targets] - hp[pulse_targets]).clamp(min=0.0)
                    actual_pulse = torch.minimum(torch.full_like(missing_hp, pulse_amount), missing_hp)

                    staging_buf.delta_hp.index_add_(0, pulse_targets, actual_pulse)
                    total_healed = float(actual_pulse.sum().item())
                    healing_done[uid] += total_healed
                    fitness[uid] += total_healed * 3.5

            # 3. Sniper: Focus Zaměření (buff na 5.0 s: extrémní zorný úhel a dosah)
            elif c_id == CLASS_SNIPER:
                ability_cooldown[uid] = 20.0
                sniper_focus_timer[uid] = 5.0
                fitness[uid] += 2.0

            # 4. Juggernaut: Fortress Shield (buff na 6.0 s: ignoruje stagger a snižuje poškození o 50%)
            elif c_id == CLASS_JUGGERNAUT:
                ability_cooldown[uid] = 22.0
                fortress_shield_timer[uid] = 6.0
                fitness[uid] += 2.5

        if new_smokes:
            new_t = torch.tensor(new_smokes, dtype=torch.float32, device=self.device)
            active_smokes_gpu = torch.cat([active_smokes_gpu, new_t], dim=0)

        return ability_cooldown, active_smokes_gpu, fortress_shield_timer, sniper_focus_timer
