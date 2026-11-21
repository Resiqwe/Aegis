"""
spatial_system.py - Prostorové dotazy, kolize se zdmi a analytický Line-of-Sight.

Využívá vysoce optimalizované maticové operace v PyTorch CUDA pro 400+ jednotek @ 180Hz.
Zahrnuje Ray-AABB Slab algoritmus pro LoS a kolizní systém s tečným skluzem.
"""

import math
import torch
from config import WORLD_W, WORLD_H


def check_line_of_sight(
    p1: torch.Tensor,
    p2: torch.Tensor,
    pillars: torch.Tensor,
    walls: torch.Tensor,
    smokes: torch.Tensor,
) -> torch.Tensor:
    """
    Přesný analytický výpočet viditelnosti (Line-of-Sight) pomocí Ray-AABB Slab algoritmu
    s podporou kouřových clon blokujících viditelnost (Smoke Grenades).
    """
    K = p1.shape[0]
    if K == 0:
        return torch.zeros((0,), dtype=torch.bool, device=p1.device)

    blocked = torch.zeros((K,), dtype=torch.bool, device=p1.device)
    d = p2 - p1
    len_sq = (d * d).sum(dim=1, keepdim=True).clamp(min=1.0)

    # 1. Kruhové pilíře
    if pillars.shape[0] > 0:
        ox = pillars[:, 0:2].unsqueeze(0)
        r = (pillars[:, 2] * 0.96).unsqueeze(0)
        f = ox - p1.unsqueeze(1)
        t = (f * d.unsqueeze(1)).sum(dim=2) / len_sq
        t = t.clamp(0.0, 1.0)
        closest = p1.unsqueeze(1) + t.unsqueeze(2) * d.unsqueeze(1)
        dist_sq = ((closest - ox) ** 2).sum(dim=2)
        blocked = blocked | (dist_sq < (r * r)).any(dim=1)

    # 2. Kouřové clony (Smoke clouds) - blokují viditelnost a sniper locky
    if smokes.shape[0] > 0:
        sox = smokes[:, 0:2].unsqueeze(0)
        sr = smokes[:, 2].unsqueeze(0)
        sf = sox - p1.unsqueeze(1)
        st = (sf * d.unsqueeze(1)).sum(dim=2) / len_sq
        st = st.clamp(0.0, 1.0)
        s_closest = p1.unsqueeze(1) + st.unsqueeze(2) * d.unsqueeze(1)
        s_dist_sq = ((s_closest - sox) ** 2).sum(dim=2)
        blocked = blocked | (s_dist_sq < (sr * sr)).any(dim=1)

    # 3. Obdélníkové zdi (Slab method)
    if walls.shape[0] > 0:
        wx1 = walls[:, 0].unsqueeze(0)
        wy1 = walls[:, 1].unsqueeze(0)
        wx2 = walls[:, 2].unsqueeze(0)
        wy2 = walls[:, 3].unsqueeze(0)

        p1x = p1[:, 0:1]
        p1y = p1[:, 1:2]
        dx = d[:, 0:1]
        dy = d[:, 1:2]

        inv_dx = torch.where(
            torch.abs(dx) > 1e-5,
            1.0 / dx,
            torch.where(dx >= 0.0, 1e6, -1e6),
        )
        inv_dy = torch.where(
            torch.abs(dy) > 1e-5,
            1.0 / dy,
            torch.where(dy >= 0.0, 1e6, -1e6),
        )

        tx1 = (wx1 - p1x) * inv_dx
        tx2 = (wx2 - p1x) * inv_dx
        tmin_x = torch.minimum(tx1, tx2)
        tmax_x = torch.maximum(tx1, tx2)

        ty1 = (wy1 - p1y) * inv_dy
        ty2 = (wy2 - p1y) * inv_dy
        tmin_y = torch.minimum(ty1, ty2)
        tmax_y = torch.maximum(ty1, ty2)

        t_enter = torch.maximum(tmin_x, tmin_y)
        t_exit = torch.minimum(tmax_x, tmax_y)

        # Přesné protnutí úsečky s AABB (žádné proklouznutí ani falešné blokace)
        wall_hit = (t_enter <= t_exit) & (t_exit >= 0.005) & (t_enter <= 0.995)
        blocked = blocked | wall_hit.any(dim=1)

    return ~blocked


class SpatialSystem:
    def __init__(self, width: int = WORLD_W, height: int = WORLD_H, device: torch.device | None = None):
        self.W = float(width)
        self.H = float(height)
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_los_diag = math.hypot(self.W, self.H)

    def check_los(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        pillars: torch.Tensor,
        walls: torch.Tensor,
        smokes: torch.Tensor,
    ) -> torch.Tensor:
        return check_line_of_sight(p1, p2, pillars, walls, smokes)

    def resolve_obstacle_collisions(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        walls: torch.Tensor,
        pillars: torch.Tensor,
        unit_radius: float = 10.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Plně maticové a bezvětvené vyřešení kolizí jednotek se zdmi a pilíři (0 CPU-GPU sync):
        - Dávková kruh-AABB detekce přes matici (N, Nw) bez Python smyček a bez .item()/.any().
        - Dávková kruh-kruh detekce přes matici (N, Np) s tečným skluzem rychlosti.
        """
        # Hranice mapy
        pos[:, 0].clamp_(unit_radius, self.W - unit_radius)
        pos[:, 1].clamp_(unit_radius, self.H - unit_radius)

        # 1. Obdélníkové stěny (AABB) - plně maticově (N, Nw)
        if walls.shape[0] > 0:
            p_x = pos[:, 0:1]  # (N, 1)
            p_y = pos[:, 1:2]
            wx1 = walls[:, 0].unsqueeze(0)  # (1, Nw)
            wy1 = walls[:, 1].unsqueeze(0)
            wx2 = walls[:, 2].unsqueeze(0)
            wy2 = walls[:, 3].unsqueeze(0)

            cx = p_x.clamp(wx1, wx2)
            cy = p_y.clamp(wy1, wy2)

            dx = p_x - cx
            dy = p_y - cy
            dist_sq = dx * dx + dy * dy

            is_inside = (p_x >= wx1) & (p_x <= wx2) & (p_y >= wy1) & (p_y <= wy2)
            is_touching = (dist_sq < (unit_radius * unit_radius)) & (~is_inside)

            # Dotyk stěny zvenku (tečný skluz + vytlačení po normále)
            d = torch.sqrt(dist_sq.clamp(min=1e-10)).clamp(min=1e-5)
            nx = dx / d
            ny = dy / d
            touch_f = is_touching.to(pos.dtype)

            touch_dx = ((cx + nx * unit_radius) - p_x) * touch_f
            touch_dy = ((cy + ny * unit_radius) - p_y) * touch_f

            v_dot_n = vel[:, 0:1] * nx + vel[:, 1:2] * ny
            into_wall_f = (is_touching & (v_dot_n < 0.0)).to(vel.dtype)
            vel[:, 0] = vel[:, 0] - (v_dot_n * nx * into_wall_f).sum(dim=1)
            vel[:, 1] = vel[:, 1] - (v_dot_n * ny * into_wall_f).sum(dim=1)

            # Jednotka uvnitř AABB stěny (vytlačení k nejbližší hraně)
            dl = p_x - wx1
            dr = wx2 - p_x
            dt = p_y - wy1
            db = wy2 - p_y
            push_h = torch.minimum(dl, dr) < torch.minimum(dt, db)
            ins_dx = torch.where(dl < dr, wx1 - unit_radius, wx2 + unit_radius) - p_x
            ins_dy = torch.where(dt < db, wy1 - unit_radius, wy2 + unit_radius) - p_y

            ins_mask_x = (is_inside & push_h).to(pos.dtype)
            ins_mask_y = (is_inside & (~push_h)).to(pos.dtype)

            pos[:, 0] = pos[:, 0] + (touch_dx + ins_dx * ins_mask_x).sum(dim=1)
            pos[:, 1] = pos[:, 1] + (touch_dy + ins_dy * ins_mask_y).sum(dim=1)

        # 2. Kruhové pilíře - plně maticově (N, Np)
        if pillars.shape[0] > 0:
            p_x = pos[:, 0:1]
            p_y = pos[:, 1:2]
            cx = pillars[:, 0].unsqueeze(0)
            cy = pillars[:, 1].unsqueeze(0)
            pr = pillars[:, 2].unsqueeze(0) + unit_radius

            dx = p_x - cx
            dy = p_y - cy
            dist_sq = dx * dx + dy * dy
            inside = dist_sq < (pr * pr)

            d = torch.sqrt(dist_sq.clamp(min=1e-10)).clamp(min=0.5)
            nx = dx / d
            ny = dy / d
            ins_f = inside.to(pos.dtype)

            pos[:, 0] = pos[:, 0] + (((cx + nx * pr) - p_x) * ins_f).sum(dim=1)
            pos[:, 1] = pos[:, 1] + (((cy + ny * pr) - p_y) * ins_f).sum(dim=1)

            v_dot_n = vel[:, 0:1] * nx + vel[:, 1:2] * ny
            into_pil_f = (inside & (v_dot_n < 0.0)).to(vel.dtype)
            vel[:, 0] = vel[:, 0] - (v_dot_n * nx * into_pil_f).sum(dim=1)
            vel[:, 1] = vel[:, 1] - (v_dot_n * ny * into_pil_f).sum(dim=1)

        return pos, vel
