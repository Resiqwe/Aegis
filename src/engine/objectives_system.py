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
