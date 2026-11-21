"""
engine/locomotion_system.py - Pohybové řízení z výstupů mozku, schopnosti, stavový registr a FOV ztráta cíle.

Bezstavový subsystém pro Fázi 2 simulačního ticku:
- Koheze čety a komunikační kanál (squad_ping)
- Vyhodnocení aktivních třídních schopností (CombatSystem.process_abilities)
- Holonomický pohyb (Surge/Sway/Stance), kolize s překážkami a hranicemi arény
- Maticový stavový registr (ArgMax State Selector pro HUD telemetrii)
- Nezávislé otáčení (Yaw), anti-káča fitness penalizace a ztráta zaměření mimo FOV
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ai_brains import CLASS_JUGGERNAUT, CLASS_MEDIC, CLASS_SNIPER

if TYPE_CHECKING:
    from engine.tick_context import TickContext
    from engine_core import RealisticBattleEngine

# Konstanty stavového registru a doktrín (shodné s engine_core)
DOCTRINE_RUSH = 1
DOCTRINE_HOLD = 2

STATE_IDLE = 0
STATE_ENGAGE = 1
STATE_COVER_RETREAT = 2
STATE_SNIPER_LOCK = 3
STATE_TRIAGE_HEAL = 4
STATE_FORTRESS_PUSH = 5


def run_locomotion(eng: RealisticBattleEngine, ctx: TickContext) -> None:
    """Vyhodnotí pohyb, rotaci, schopnosti, stavový registr a ztrátu cíle mimo FOV."""
    dt = ctx.dt
    N = ctx.N
    alive = ctx.alive
    alive_idx = ctx.alive_idx
    K = ctx.K
    wall_force = ctx.wall_force
    dir_enemy = ctx.dir_enemy
    dist_enemy = ctx.dist_enemy
    target_enemy = ctx.target_enemy
    medic_has_patient = ctx.medic_has_patient
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
    squad_force = torch.zeros((N, 2), dtype=torch.float32, device=eng.device)
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
    brain_surge = brain_out[:, 0]    # +1 vpřed, -1 plná zpátečka / couvání
    brain_sway = brain_out[:, 1]     # +1 vpravo, -1 vlevo
    brain_yaw = brain_out[:, 2]      # rotace těla nezávisle na směru pohybu
    brain_stance = brain_out[:, 4]   # +1 Sprint (+30% rychlost, horší kryt), -1 Plížení (poloviční rychlost, max kryt)
    brain_ability = brain_out[:, 5]  # > 0.60 spouští aktivní třídní schopnost
    brain_ping = brain_out[:, 6]     # taktický signál vysílaný spojencům v četě

    # Aktualizace komunikačního kanálu čety
    eng.squad_ping = torch.lerp(eng.squad_ping, brain_ping * alive.float(), 0.35)

    # Vyhodnocení třídních schopností (Akce 5: Ability) přes CombatSystem
    (
        eng.ability_cooldown,
        eng.active_smokes_gpu,
        eng.fortress_shield_timer,
        eng.sniper_focus_timer,
    ) = eng.combat.process_abilities(
        dt=dt,
        brain_ability=brain_ability,
        alive=alive,
        classes=eng.classes,
        teams=eng.teams,
        pos=eng.pos,
        hp=eng.hp,
        class_max_hp=eng.class_max_hp,
        ability_cooldown=eng.ability_cooldown,
        active_smokes_gpu=eng.active_smokes_gpu,
        fortress_shield_timer=eng.fortress_shield_timer,
        sniper_focus_timer=eng.sniper_focus_timer,
        staging_buf=eng.staging_buf,
        healing_done=eng.healing_done,
        fitness=eng.fitness,
        active_heals_p1=[],
        active_heals_p2=[],
        faction_techs=eng.faction_active_techs,
        visual_mode=eng.visual_mode,
        particles=eng.particles,
    )

    # Holonomická transformace tělesného vektoru (Surge & Sway) do globálních souřadnic
    # Agent se může pohybovat libovolným směrem v 360° a zároveň mířit jinam!
    brain_mv_x = brain_surge * cos_th + brain_sway * (-sin_th)
    brain_mv_y = brain_surge * sin_th + brain_sway * cos_th
    brain_world_vec = torch.stack([brain_mv_x, brain_mv_y], dim=1)

    # Spojení tahu s odpudivostí překážek a kohezí
    move_raw = brain_world_vec * 1.35 + squad_force * 0.18 + wall_force * 1.60

    # Odpočet Stagger timeru (ochrnutí hybnosti po těžkém zásahu)
    eng.stagger_timer = (eng.stagger_timer - dt).clamp(min=0.0)
    is_staggered = eng.stagger_timer > 0.0

    # ==============================================================
    # MATICOVÝ STAVOVÝ REGISTR (ArgMax State Selector)
    # Tenzorové vyhodnocení stavů bez vnořených větví if/else
    # Stavy: 0: IDLE, 1: ENGAGE, 2: COVER_RETREAT, 3: SNIPER_LOCK, 4: TRIAGE_HEAL, 5: FORTRESS_PUSH
    # ==============================================================
    utility = torch.zeros((N, 6), dtype=torch.float32, device=eng.device)
    u_classes = eng.classes.long()

    # Stav 0: IDLE (výchozí základní váha)
    utility[:, STATE_IDLE] = 0.50

    # Stav 1: ENGAGE (čelní cíl v dosahu, nízké potlačení)
    has_combat_target = (dist_enemy <= eng.class_range * 1.35) & eng.has_los_target & (target_enemy >= 0) & alive
    enemy_ang = torch.atan2(dir_enemy[:, 1], dir_enemy[:, 0])
    aim_diff = (enemy_ang - eng.heading + math.pi) % (2.0 * math.pi) - math.pi
    engage_utility = torch.where(
        has_combat_target & (torch.abs(aim_diff) < math.radians(45)),
        1.20 + (1.0 - eng.suppression) * 0.80,
        0.0,
    )
    utility[:, STATE_ENGAGE] = engage_utility

    # Stav 2: COVER_RETREAT (těžké potlačení nebo poškození mimo kryt -> pud sebezáchovy)
    retreat_utility = torch.where(
        (~eng.in_cover) & ((eng.suppression > 0.40) | (eng.hp < eng.class_max_hp * 0.35)),
        1.60 + eng.suppression * 0.90,
        0.0,
    )
    utility[:, STATE_COVER_RETREAT] = retreat_utility

    # Stav 3: SNIPER_LOCK (Sniper s čistým LoS raycastem zastavuje pohyb a fixuje zbraň)
    is_sniper_cls = u_classes == CLASS_SNIPER
    sniper_lock_utility = torch.where(
        is_sniper_cls & has_combat_target & (eng.ammo_in_mag > 0),
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
    eng.unit_state = torch.argmax(utility, dim=1).int()

    # Zero-Script AI: Pohyb je plně řízen výstupem neuronové sítě (Surge, Sway, Stance)
    # s přihlédnutím k fyzikálnímu odpuzování od stěn a kohezi formace.
    effective_move_raw = move_raw

    u_doc = eng.faction_doctrine_tensor[eng.teams.long()]
    doc_speed_mod = torch.where(u_doc == DOCTRINE_RUSH, 1.12, torch.where(u_doc == DOCTRINE_HOLD, 0.90, 1.0))
    # Stance modifikátor rychlosti: +1.0 = Sprint (1.30x), -1.0 = Plížení (0.55x)
    stance_speed_mod = torch.where(brain_stance > 0.3, 1.30, torch.where(brain_stance < -0.3, 0.55, 1.0))
    speed_mult = torch.where(
        is_staggered,
        torch.full_like(doc_speed_mod, 0.15),
        (1.0 - eng.suppression * 0.40) * doc_speed_mod * stance_speed_mod,
    )

    v_norm = torch.norm(effective_move_raw, dim=1, keepdim=True).clamp(min=1.0)
    desired_vel = (effective_move_raw / v_norm) * speed_mult.unsqueeze(1) * eng.class_speed.unsqueeze(1)

    eng.vel = torch.lerp(eng.vel, desired_vel, 0.26) * alive.unsqueeze(1).float()
    eng.pos += eng.vel

    eng.pos, eng.vel = eng.spatial.resolve_obstacle_collisions(eng.pos, eng.vel, eng.walls_gpu, eng.pillars_gpu, 10.0)
    eng.pos[:, 0].clamp_(25.0, eng.W - 25.0)
    eng.pos[:, 1].clamp_(25.0, eng.H - 25.0)

    # Nezávislé otáčení těla a zaměřování (Yaw):
    # Rotace je řízena mozkem (brain_yaw). V boji s přímou viditelností poskytuje jemnou taktickou asistenci.
    omega = torch.where(has_combat_target, aim_diff * 0.45 + brain_yaw * 0.20, brain_yaw * 0.45)
    eng.heading = (eng.heading + omega * alive.float() + math.pi) % (2.0 * math.pi) - math.pi

    # ANTI-KÁČA FITNESS PENALIZACE ZA ZBYTEČNOU ÚHLOVOU KINETIKU:
    # Pokud se agent bezcílně točí na místě bez lineárního pohybu, okamžitě padá jeho fitness!
    lin_speed = torch.norm(eng.vel, dim=1)
    spin_penalty = torch.where((lin_speed < 0.20) & (~has_combat_target) & alive, omega.abs() * 0.18, torch.zeros_like(omega))
    eng.fitness -= spin_penalty

    # Okamžitá ztráta zaměření cíle mimo FOV kužel (100 % bezvětveně na GPU bez .any()/nonzero)
    has_tar = (target_enemy >= 0) & (target_enemy < eng.N) & alive
    tar_safe = target_enemy.clamp(0, max(0, N - 1))
    tar_v = eng.pos[tar_safe] - eng.pos
    tar_d = torch.norm(tar_v, dim=1)
    tar_a = torch.atan2(tar_v[:, 1], tar_v[:, 0])
    tar_diff = (tar_a - eng.heading + math.pi) % (2.0 * math.pi) - math.pi
    lost_fov = has_tar & (tar_d > 75.0) & (torch.abs(tar_diff) >= (math.radians(140) * 0.5))

    target_enemy = torch.where(lost_fov, -1, target_enemy)
    dist_enemy = torch.where(lost_fov, 9999.0, dist_enemy)
    dir_enemy = torch.where(lost_fov.unsqueeze(1), 0.0, dir_enemy)
    eng.has_los_target = eng.has_los_target & (~lost_fov)

    eng.last_target_enemy = target_enemy

    # Aktualizace TickContextu pro navazující Combat/Weapons fázi
    ctx.target_enemy = target_enemy
    ctx.dist_enemy = dist_enemy
    ctx.dir_enemy = dir_enemy
    ctx.has_target = has_combat_target
    ctx.tar_diff = tar_diff
