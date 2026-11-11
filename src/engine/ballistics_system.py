"""
engine/ballistics_system.py - Aegis 2.0 Vektorový balistický subsystém (PyTorch CUDA)
Inspirováno modelem TacticalBallisticsEngine od kolegyně, optimalizováno pro 400+ jednotek @ 180Hz.

Klíčové mechaniky:
1. Úhlová efektivita pancíře (Dot-product penetrace):
   - dot = shot_dir_x * facing_x + shot_dir_y * facing_y
   - Čelní zásah (dot < -0.5): 100 % pancíř.
   - Boční zásah (|dot| <= 0.5): 50 % pancíř.
   - Zásah do zad (dot > 0.5): 15 % pancíř (Kritický zásah / Flanking).
2. Zónový rozpad zásahu s vazbou na poloviční/plný kryt:
   - ZONE_HEAD (0): 2.5x násobič zranění.
   - ZONE_TORSO (1): 1.0x základ.
   - ZONE_ARMS (2): 0.55x zranění.
   - ZONE_LEGS (3): 0.65x zranění + těžký Mobility Stagger.
   - Při krytu (in_cover) jsou nohy a spodní trup chráněny, zásahy jdou do horní části.
3. Kinetický Ragdoll, Knockback & Stagger:
   - Lineární kinetický impuls delta_v = p_bullet / mass aplikován přímo do tenzoru vel.
   - Stagger: těžké zásahy redukují rychlost a rotaci jednotky na 0.15s.
4. Průrazová balistika (Penetration Tracers):
   - Sniper s těžkou zbraní (AP=65) po proražení cíle pokračuje v letu.
"""

import math
import torch
import numpy as np

# Konstanty zón těla
ZONE_HEAD = 0
ZONE_TORSO = 1
ZONE_ARMS = 2
ZONE_LEGS = 3

ZONE_MULTIPLIERS = [2.50, 1.00, 0.55, 0.65]

# Třídní balistické specifikace: [Grunt, Medic, Sniper, Juggernaut]
# AP (průbojnost plátů), Base Armor, Kinetic Energy (Joulů pro knockback), Unit Mass
CLASS_AP = [18.0, 10.0, 65.0, 26.0]
CLASS_BASE_ARMOR = [12.0, 8.0, 10.0, 32.0]
CLASS_KINETIC_ENERGY = [45.0, 22.0, 120.0, 95.0]
CLASS_MASS = [1.0, 0.9, 0.85, 3.2]

# Kinetické prahy pro stagger: Juggernaut odolá všemu pod 95 J, ostatní pod 40 J
CLASS_STAGGER_THRESHOLDS = [40.0, 35.0, 35.0, 95.0]
BASE_STAGGER_TIME = 0.22


class VectorizedBallisticsSystem:
    def __init__(self, device: torch.device):
        self.device = device
        self.ap_tensor = torch.tensor(CLASS_AP, dtype=torch.float32, device=device)
        self.armor_tensor = torch.tensor(CLASS_BASE_ARMOR, dtype=torch.float32, device=device)
        self.kinetic_tensor = torch.tensor(CLASS_KINETIC_ENERGY, dtype=torch.float32, device=device)
        self.mass_tensor = torch.tensor(CLASS_MASS, dtype=torch.float32, device=device)
        self.stagger_thresh_tensor = torch.tensor(CLASS_STAGGER_THRESHOLDS, dtype=torch.float32, device=device)
        self.zone_mults_tensor = torch.tensor(ZONE_MULTIPLIERS, dtype=torch.float32, device=device)

    def resolve_ballistic_hits(
        self,
        valid_s: torch.Tensor,
        valid_t: torch.Tensor,
        is_hit: torch.Tensor,
        shooter_pos: torch.Tensor,
        target_pos: torch.Tensor,
        target_heading: torch.Tensor,
        shooter_classes: torch.Tensor,
        target_classes: torch.Tensor,
        target_in_cover: torch.Tensor,
        has_kinetic_shield: torch.Tensor,
        base_dmg: torch.Tensor,
    ) -> dict:
        """
        Vektorově vyhodnotí balistiku pro celou dávku střel najednou:
        - Detekce zasažené zóny s ohledem na kryt
        - Úhel dopadu a efektivní tloušťka pancíře
        - Výsledné poškození po průrazu / absorpci
        - Kinetický impuls (knockback) a stagger
        """
        K = valid_s.numel()
        if K == 0:
            return {
                "final_damage": torch.zeros((0,), device=self.device),
                "knockback_x": torch.zeros((0,), device=self.device),
                "knockback_y": torch.zeros((0,), device=self.device),
                "stagger_mask": torch.zeros((0,), dtype=torch.bool, device=self.device),
                "hit_zones": torch.zeros((0,), dtype=torch.long, device=self.device),
                "shield_absorbed": torch.zeros((0,), dtype=torch.bool, device=self.device),
                "headshot_mask": torch.zeros((0,), dtype=torch.bool, device=self.device),
            }

        s_cls = shooter_classes.long()
        t_cls = target_classes.long()

        # 1. Směrový vektor letu střely
        diff = target_pos - shooter_pos
        dist = torch.norm(diff, dim=1).clamp(min=1e-4)
        dir_x = diff[:, 0] / dist
        dir_y = diff[:, 1] / dist

        # 2. Úhel dopadu vůči otočení cíle (Dot Product)
        # target_heading: úhel v radiánech
        facing_x = torch.cos(target_heading)
        facing_y = torch.sin(target_heading)
        # dot = 1.0 (střela letí ze zad do stejného směru), dot = -1.0 (čelní střet hlava proti hlavě)
        dot = dir_x * facing_x + dir_y * facing_y

        # Úhlový faktor pancíře: čelo (-1.0) -> 1.0, záda (+1.0) -> 0.15
        angle_factor = torch.clamp((-dot + 1.0) * 0.5, min=0.15, max=1.0)
        target_armor = self.armor_tensor[t_cls] * angle_factor

        # 3. Zónový rozpad zásahu s ohledem na kryt
        roll_zone = torch.rand(K, device=self.device)
        in_cov = target_in_cover.bool()

        # Pokud je v krytu: nohy jsou skryté za překážkou (0 % zásah), hlava 25 %, trup 50 %, paže 25 %
        zone_cov = torch.where(roll_zone < 0.25, ZONE_HEAD,
                   torch.where(roll_zone < 0.75, ZONE_TORSO, ZONE_ARMS))

        # Na volném prostranství: hlava 12 %, trup 50 %, paže 18 %, nohy 20 %
        zone_open = torch.where(roll_zone < 0.12, ZONE_HEAD,
                    torch.where(roll_zone < 0.62, ZONE_TORSO,
                    torch.where(roll_zone < 0.80, ZONE_ARMS, ZONE_LEGS)))

        hit_zones = torch.where(in_cov, zone_cov, zone_open)
        zone_multipliers = self.zone_mults_tensor[hit_zones]

        # 4. Průbojnost (AP) vs Efektivní pancíř (AR)
        bullet_ap = self.ap_tensor[s_cls]
        delta_ap = bullet_ap - target_armor

        # Pokud AP překročí pancíř: plná penetrace + 2% bonus za každý přebytečný bod AP
        pen_over = 1.0 + (delta_ap * 0.02).clamp(min=0.0, max=0.45)
        # Pokud pancíř odolá: tupé trauma (deformace plátu)
        pen_under = torch.clamp((bullet_ap / target_armor.clamp(min=1.0)) * 0.60, min=0.18, max=0.90)
        penetration_factor = torch.where(delta_ap >= 0.0, pen_over, pen_under)

        # 5. Základní poškození modifikované zónou a průrazem
        raw_damage = (base_dmg * zone_multipliers * penetration_factor) * is_hit.float()

        # 6. Absorpce kinetickým štítem (Power-Up TECH_SHIELDS)
        shield_active = has_kinetic_shield[valid_t] & is_hit
        shield_absorbed = shield_active.clone()
        final_damage = torch.where(shield_active, torch.zeros_like(raw_damage), raw_damage)

        # 7. Kinetický impuls (Knockback) a Stagger se striktním prahem podle třídy
        kin_energy = self.kinetic_tensor[s_cls]
        target_mass = self.mass_tensor[t_cls]
        target_stagger_thresh = self.stagger_thresh_tensor[t_cls]

        # Tenzorové filtrování efektivní energie: Juggernaut ignoruje vše pod 95J!
        effective_energy = torch.where(
            (t_cls == 3) & (kin_energy < 95.0), # 3 == CLASS_JUGGERNAUT
            torch.zeros_like(kin_energy),
            kin_energy
        )

        # Síla impulzu ve směru letu střely (v px/s přepočteno na tick)
        impulse_mag = (effective_energy / target_mass) * 0.045
        apply_kb = is_hit & (~shield_active)
        kb_x = torch.where(apply_kb, dir_x * impulse_mag, torch.zeros_like(dir_x))
        kb_y = torch.where(apply_kb, dir_y * impulse_mag, torch.zeros_like(dir_y))

        # Stagger: zásah do nohou nebo kinetická hybnost přesahující specifický práh třídy
        is_leg_hit = hit_zones == ZONE_LEGS
        stagger_triggered = (effective_energy > target_stagger_thresh) | is_leg_hit
        stagger_mask = apply_kb & stagger_triggered
        stagger_duration = torch.where(stagger_mask, BASE_STAGGER_TIME, 0.0)
        headshot_mask = apply_kb & (hit_zones == ZONE_HEAD)

        # Yield potlačení palbou (Suppression yield) generovaný balistickým zásahem
        # Juggernaut má vysokou odolnost vůči potlačení (0.16 vs 0.35)
        suppression_yield = torch.where(t_cls == 3, 0.16, 0.35) * is_hit.float()

        return {
            "final_damage": final_damage,
            "knockback_x": kb_x,
            "knockback_y": kb_y,
            "stagger_mask": stagger_mask,
            "stagger_duration": stagger_duration,
            "suppression_yield": suppression_yield,
            "hit_zones": hit_zones,
            "shield_absorbed": shield_absorbed,
            "headshot_mask": headshot_mask,
            "flank_hit": apply_kb & (dot > 0.40),
        }
