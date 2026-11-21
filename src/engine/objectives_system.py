"""
objectives_system.py - Správa strategických nodů, zničitelných bran a vyhodnocení vítězství.
"""

import threading
import numpy as np
import torch
from config import CLASS_JUGGERNAUT, MAX_OPERATION_DURATION


class ObjectivesSystem:
    def __init__(self, device: torch.device):
        self.device = device
        self._lock = threading.RLock()

    def update_control_nodes(
        self,
        dt: float,
        nodes_gpu: torch.Tensor,
        node_control_points: torch.Tensor,
        pos: torch.Tensor,
        teams: torch.Tensor,
        alive: torch.Tensor,
        n_factions: int = 4,
    ) -> torch.Tensor:
        """Obsazování strategických nodů podle převahy jednotek v rádiusu."""
        if nodes_gpu.shape[0] == 0:
            return nodes_gpu

        n_nodes = nodes_gpu.shape[0]
        for ni in range(n_nodes):
            nx, ny, nr = nodes_gpu[ni, 0], nodes_gpu[ni, 1], nodes_gpu[ni, 2]
            dists = torch.hypot(pos[:, 0] - nx, pos[:, 1] - ny)
            inside = (dists <= nr) & alive

            if inside.any():
                in_teams = teams[inside].long()
                cnts = torch.bincount(in_teams, minlength=n_factions)
                max_cnt = int(torch.max(cnts).item())
                winners = (cnts == max_cnt).nonzero().squeeze(1)

                if len(winners) == 1 and max_cnt > 0:
                    dominant_team = int(winners[0].item())
                    cur_owner = int(nodes_gpu[ni, 3].item())
                    cur_prog = float(nodes_gpu[ni, 4].item())

                    if cur_owner == dominant_team:
                        nodes_gpu[ni, 4] = min(1.0, cur_prog + dt * 0.45)
                    else:
                        new_prog = cur_prog - dt * 0.55
                        if new_prog <= 0.0:
                            nodes_gpu[ni, 3] = float(dominant_team)
                            nodes_gpu[ni, 4] = 0.05
                        else:
                            nodes_gpu[ni, 4] = new_prog

            owner = int(nodes_gpu[ni, 3].item())
            prog = float(nodes_gpu[ni, 4].item())
            if owner >= 0 and prog >= 0.85:
                node_control_points[owner] += dt * 1.5

        return nodes_gpu

    def update_gates(
        self,
        dt: float,
        gates: list[dict],
        gate_hp: torch.Tensor,
        pos: torch.Tensor,
        teams: torch.Tensor,
        classes: torch.Tensor,
        alive: torch.Tensor,
    ) -> list[int]:
        """
        Obléhání zničitelných bran: útočníci v blízkosti poškozují bránu.
        Vrací seznam indexů bran nově zničených v tomto kroku.
        """
        if not gates or gate_hp.numel() == 0:
            return []

        destroyed_now = []
        for gi, gate in enumerate(gates):
            if gate.get("destroyed", False) or gate_hp[gi] <= 0.0:
                continue

            x1, y1, x2, y2 = gate["rect"]
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            reach = max(x2 - x1, y2 - y1) * 0.5 + 160.0
            d2 = (pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2
            attackers = alive & (teams != gate["team"]) & (d2 <= reach * reach)
            if not attackers.any():
                continue

            att_classes = classes[attackers].long()
            att_mult = torch.where(att_classes == CLASS_JUGGERNAUT, 3.0, 1.0)
            dmg = float((att_mult * 18.0).sum().item()) * dt
            new_hp = max(0.0, float(gate_hp[gi].item()) - dmg)
            gate_hp[gi] = new_hp
            gate["hp"] = new_hp
            if new_hp <= 0.0:
                gate["destroyed"] = True
                destroyed_now.append(gi)

        return destroyed_now

    def check_battle_status(
        self,
        operation_time_sec: float,
        teams: torch.Tensor,
        alive: torch.Tensor,
        initial_team_counts: list[int],
        kills_per_faction: torch.Tensor,
        node_control_points: torch.Tensor,
        n_factions: int = 4,
        max_duration: float = MAX_OPERATION_DURATION,
    ) -> tuple[bool, int, str]:
        """
        Vyhodnotí stav bitvy:
        Vrací (is_finished, winner_team, reason).
        Tie-breaking: Kills * 3.0 + NodePoints * 2.5 + AliveUnits
        """
        with self._lock:
            alive_teams = teams[alive].long()
            cnts = torch.bincount(alive_teams, minlength=n_factions).cpu().numpy()
            alive_counts = [int(c) for c in cnts]
            combat_effective = []

            for f in range(n_factions):
                c = alive_counts[f]
                init_c = initial_team_counts[f] if f < len(initial_team_counts) else 100
                if c >= max(3, int(init_c * 0.22)):
                    combat_effective.append(f)

            if len(combat_effective) == 1 and operation_time_sec > 4.0:
                return True, combat_effective[0], "KAPITULACE ODPŮRCŮ (Bojeschopnost < 22 %)"

            if len(combat_effective) == 0 and operation_time_sec > 4.0:
                winner = int(np.argmax(alive_counts))
                return True, winner, "TOTÁLNÍ VYČERPÁNÍ SIL"

            if operation_time_sec >= max_duration:
                nc_cpu = node_control_points.cpu().numpy()
                k_cpu = kills_per_faction.cpu().numpy()
                scores = [
                    k_cpu[f] * 3.0 + nc_cpu[f] * 2.5 + alive_counts[f]
                    for f in range(n_factions)
                ]
                winner = int(np.argmax(scores))
                return True, winner, f"VYPRŠENÍ OPERAČNÍHO ČASU ({int(max_duration)}s)"

            return False, -1, ""


def run_post_commit_and_objectives(eng, ctx) -> None:
    """
    Fáze po atomickém commitu (_atomic_commit_pipeline):
    - Zaměřovací lasery Sniperů (v visual_mode)
    - Anti-kemping penalizace Sniperů
    - Aktualizace částic výbuchů a kouře
    - Plně vektorizované obsazování kontrolních uzlů a bodování na GPU
    """
    import math
    from ai_brains import CLASS_GRUNT, CLASS_SNIPER

    dt = ctx.dt
    alive = ctx.alive
    alive_idx = ctx.alive_idx
    K = ctx.K
    dist_enemy = ctx.dist_enemy
    target_enemy = ctx.target_enemy

    eng.last_target_enemy = target_enemy

    # Zaměřovací lasery Sniperů – POUZE pokud mají přímou viditelnost a míří na cíl dopředu!
    aiming_snipers = (
        alive
        & (eng.classes == CLASS_SNIPER)
        & (dist_enemy <= eng.class_range)
        & (dist_enemy > 10.0)
        & eng.has_los_target
    )
    if not eng.visual_mode:
        aiming_snipers = torch.zeros_like(aiming_snipers)
    aim_idx = torch.where(aiming_snipers)[0]
    if aim_idx.numel() > 0:
        a_tar = target_enemy[aim_idx]
        valid_tar_snipers = a_tar >= 0
        if valid_tar_snipers.any():
            s_valid = aim_idx[valid_tar_snipers]
            t_valid = a_tar[valid_tar_snipers]
            l_vec = eng.pos[t_valid] - eng.pos[s_valid]
            l_ang = torch.atan2(l_vec[:, 1], l_vec[:, 0])
            l_err = (l_ang - eng.heading[s_valid] + math.pi) % (2.0 * math.pi) - math.pi
            facing = torch.abs(l_err) < math.radians(50)
            if facing.any():
                s_face = s_valid[facing]
                t_face = t_valid[facing]
                a_lim = min(s_face.numel(), 20)
                a_sel = s_face[:a_lim]
                a_tar_sel = t_face[:a_lim]
                eng.sniper_lasers = torch.cat([eng.pos[a_sel], eng.pos[a_tar_sel]], dim=1)
            else:
                eng.sniper_lasers = torch.zeros((0, 4), dtype=torch.float32, device=eng.device)
        else:
            eng.sniper_lasers = torch.zeros((0, 4), dtype=torch.float32, device=eng.device)
    else:
        eng.sniper_lasers = torch.zeros((0, 4), dtype=torch.float32, device=eng.device)

    # Sniper anti-kemping penalizace (pokud sniper stojí déle než 8s na místě, ztrácí fitness)
    is_sniper = (eng.classes == CLASS_SNIPER) & alive
    is_standing = torch.norm(eng.vel, dim=1) < 0.20
    eng.sniper_camp_timer = torch.where(
        is_sniper & is_standing, eng.sniper_camp_timer + dt, torch.zeros_like(eng.sniper_camp_timer)
    )
    camp_penalty = (eng.sniper_camp_timer > 8.0) & is_sniper
    if camp_penalty.any():
        eng.fitness[camp_penalty] -= 1.8 * dt

    # Aktualizace částic výbuchů a kouře
    alive_particles = []
    for p in eng.particles:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["vx"] *= 0.92
        p["vy"] *= 0.92
        p["life"] -= dt * 2.2
        if p["life"] > 0:
            alive_particles.append(p)
    eng.particles = alive_particles

    # Obsazování uzlů (plně vektorizovaně na GPU v matici (K, M) bez .item() ve smyčce)
    if eng.nodes_gpu.shape[0] > 0 and (eng.tick_counter % 6 == 0) and K > 0:
        p_alive = eng.pos[alive_idx]
        t_alive = eng.teams[alive_idx].long()
        step_scale = 6.0

        # 1. Maticový výpočet přítomnosti jednotek v uzlech: tvar (K, M)
        diff = p_alive.unsqueeze(1) - eng.nodes_gpu[:, :2].unsqueeze(0)   # (K, M, 2)
        dists_sq = (diff * diff).sum(dim=-1)                              # (K, M)
        radii_sq = eng.nodes_gpu[:, 2].unsqueeze(0) ** 2                  # (1, M)
        in_node = dists_sq <= radii_sq                                    # (K, M) bool

        # 2. Počty jednotek jednotlivých frakcí v každém uzlu: tvar (M, 4)
        team_one_hot = torch.nn.functional.one_hot(t_alive, num_classes=4).to(torch.float32)
        cnts = in_node.to(torch.float32).transpose(0, 1) @ team_one_hot   # (M, 4)

        cnt_dom, dom_t = torch.max(cnts, dim=1)                           # (M,), (M,)
        cnt_other = cnts.sum(dim=1) - cnt_dom                             # (M,)
        has_majority = cnt_dom > cnt_other                                # (M,) bool

        n_owner = eng.nodes_gpu[:, 3].long()                              # (M,)
        n_cap = eng.nodes_gpu[:, 4].clone()                               # (M,)

        # 3. Vektorový výpočet fitness pro obsazující jednotky (+250 % bonus pro CLASS_GRUNT)
        is_grunt = eng.classes[alive_idx] == CLASS_GRUNT
        bonus_mult = torch.where(is_grunt, 2.5, 1.0).unsqueeze(1)         # (K, 1)
        is_cap_unit = in_node & has_majority.unsqueeze(0) & (t_alive.unsqueeze(1) == dom_t.unsqueeze(0))

        same_owner = n_owner == dom_t                                     # (M,)
        reinforcing = has_majority & same_owner & (n_cap < 1.0)           # (M,)
        neutralizing = has_majority & (~same_owner)                       # (M,)

        node_fit_rate = torch.where(
            reinforcing,
            0.45 * step_scale,
            torch.where(neutralizing, 0.65 * step_scale, 0.0),
        ).unsqueeze(0)                                                    # (1, M)

        cap_fitness_gain = (is_cap_unit.to(torch.float32) * bonus_mult * node_fit_rate).sum(dim=1)
        eng.fitness[alive_idx] += cap_fitness_gain

        # 4. Aktualizace progresu obsazení (n_cap) a vlastníka (n_owner) na GPU
        n_cap = torch.where(reinforcing, torch.clamp(n_cap + 0.04 * step_scale, max=1.0), n_cap)
        n_cap = torch.where(neutralizing, n_cap - 0.05 * step_scale, n_cap)

        captured_mask = neutralizing & (n_cap <= 0.0)                     # (M,) bool
        n_owner = torch.where(captured_mask, dom_t, n_owner)
        n_cap = torch.where(captured_mask, torch.full_like(n_cap, 0.12), n_cap)

        eng.nodes_gpu[:, 3] = n_owner.to(torch.float32)
        eng.nodes_gpu[:, 4] = n_cap

        # 5. Zápis události do combat_events pouze ve vzácném tiku změny vlastníka
        if captured_mask.any():
            cap_indices = torch.where(captured_mask)[0].cpu().tolist()
            dom_teams_cpu = dom_t[captured_mask].cpu().tolist()
            sec_t = int(eng.operation_time_sec)
            meta_names = ("ALPHA", "OMEGA", "SOLAR", "PHANTOM")
            node_names = ("UZEL ALFA", "UZEL BRAVO", "UZEL CHARLIE", "UZEL DELTA")

            for idx_node, d_team in zip(cap_indices, dom_teams_cpu):
                if len(eng.combat_events) > 35:
                    eng.combat_events.pop(0)
                n_label = node_names[idx_node] if idx_node < 4 else f"UZEL #{idx_node + 1}"
                eng.combat_events.append({
                    "time": eng.operation_time_sec,
                    "text": f"[{sec_t:02d}s] {meta_names[d_team]} obsadil {n_label}!",
                    "team": d_team,
                })

        # 6. Pasivní bodování a obranná fitness bez CPU větvení (n_owner >= 0 a n_cap >= 0.7)
        held_mask = (n_owner >= 0) & (n_cap >= 0.7)                       # (M,) bool
        safe_owners = torch.clamp(n_owner, min=0)
        pts_add = torch.where(held_mask, 0.08 * step_scale, 0.0).to(eng.node_control_points.dtype)
        eng.node_control_points.index_add_(0, safe_owners, pts_add)

        is_defending = in_node & held_mask.unsqueeze(0) & (t_alive.unsqueeze(1) == n_owner.unsqueeze(0))
        def_nodes_count = is_defending.sum(dim=1).to(torch.float32)       # (K,)
        eng.fitness[alive_idx] += def_nodes_count * (0.25 * step_scale)
