import math
import numpy as np
from collections import deque
import torch

from src.config import (
    CLASS_GRUNT, CLASS_MEDIC, CLASS_SNIPER, CLASS_JUGGERNAUT,
    CLASS_COSTS, CLASS_NAMES, UNLOCK_XP, COMMAND_RANKS,
    TECH_NANITES, TECH_RADAR, TECH_SHIELDS, TECH_OVERCLOCK, TECH_AEROSOL,
    TECH_NAMES, TECH_COSTS,
    SENSOR_NAMES, ACTION_NAMES, SENSOR_COUNT, ACTION_COUNT
)


def tactical_brain_forward(
    x: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    W3: torch.Tensor,
    b3: torch.Tensor,
) -> torch.Tensor:
    """
    2-vrstvý hierarchický MLP:
    x (B, 14) -> H1: Percepce (B, 64) -> H2: Taktická syntéza (B, 64) -> 7 Akcí (B, 7)
    Výstupy:
    0: Surge (tanh [-1..1]) - Plyn vpřed / couvání
    1: Sway (tanh [-1..1]) - Úkrok vpravo / vlevo
    2: Yaw (tanh [-1..1]) - Otáčení těla a zbraně
    3: Trigger (sigmoid [0..1]) - >0.58 spouští střelbu
    4: Stance (tanh [-1..1]) - +1 Sprint / -1 Zakleknutí (kryt)
    5: Ability (sigmoid [0..1]) - >0.60 aktivuje třídní schopnost
    6: Squad Ping (tanh [-1..1]) - Komunikační signál četě
    """
    x_3d = x.unsqueeze(1)
    h1 = torch.tanh(torch.bmm(x_3d, W1) + b1)
    h2 = torch.tanh(torch.bmm(h1, W2) + b2)
    raw_out = torch.bmm(h2, W3) + b3

    surge_sway_yaw = torch.tanh(raw_out[:, :, 0:3])
    trigger = torch.sigmoid(raw_out[:, :, 3:4])
    stance = torch.tanh(raw_out[:, :, 4:5])
    ability = torch.sigmoid(raw_out[:, :, 5:6])
    ping = torch.tanh(raw_out[:, :, 6:7])

    out = torch.cat([surge_sway_yaw, trigger, stance, ability, ping], dim=2)
    return out.squeeze(1)


class FactionGeneralBrain:
    """
    Strategický mozek frakce (Generál).
    Optimalizuje nákup armády a technologií za body podle ROI (Return on Investment).
    """
    def __init__(self, faction_id: int, device: torch.device):
        self.faction_id = faction_id
        self.device = device
        self.xp = 0.0
        self.level = 0
        self.total_campaign_wins = 0

        # Vyvážené počáteční váhy pro všechny 4 třídy
        self.composition_weights = torch.tensor([1.0, 0.7, 0.6, 0.5], dtype=torch.float32, device=device)
        self.class_roi_ema = [0.0, 0.0, 0.0, 0.0]

        # Technologie: Nanity, Radar, Štíty, Overclock, Aerosol
        self.tech_weights = torch.tensor([0.6, 0.5, 0.7, 0.6, 0.5], dtype=torch.float32, device=device)
        self.tech_roi_ema = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.active_techs: list[int] = []

    def update_level(self) -> bool:
        old_level = self.level
        for lvl, req in enumerate(UNLOCK_XP):
            if self.xp >= req:
                self.level = lvl
        return self.level > old_level

    def get_rank_title(self) -> str:
        idx = min(self.level, len(COMMAND_RANKS) - 1)
        return COMMAND_RANKS[idx]

    def get_unlocked_classes(self) -> list[int]:
        # Všechny 4 třídy jsou dostupné od začátku pro maximální adaptabilitu
        return [CLASS_GRUNT, CLASS_MEDIC, CLASS_SNIPER, CLASS_JUGGERNAUT]

    def get_unlocked_techs(self) -> list[int]:
        # Všechny technologie dostupné pro nákup
        return list(range(len(TECH_NAMES)))

    def allocate_full_budget(self, total_budget: int = 250) -> tuple[list[int], list[int]]:
        unlocked_techs = self.get_unlocked_techs()
        bought_techs = []
        rem_budget = total_budget

        max_tech_budget = int(total_budget * 0.30)
        tech_spent = 0

        if unlocked_techs and max_tech_budget >= 15:
            t_probs = torch.sigmoid(self.tech_weights[unlocked_techs]).cpu().numpy()
            sorted_indices = sorted(range(len(unlocked_techs)), key=lambda k: t_probs[k], reverse=True)
            for idx in sorted_indices:
                t_id = unlocked_techs[idx]
                cost = TECH_COSTS[t_id]
                if t_probs[idx] > 0.45 and (tech_spent + cost) <= max_tech_budget and rem_budget >= cost:
                    bought_techs.append(t_id)
                    tech_spent += cost
                    rem_budget -= cost

        self.active_techs = bought_techs
        army = self.allocate_army_budget(total_budget=rem_budget)
        return army, bought_techs

    def allocate_army_budget(self, total_budget: int = 100) -> list[int]:
        unlocked = self.get_unlocked_classes()
        probs = self.get_class_probabilities()

        army = []
        rem_budget = total_budget

        min_cost = min(CLASS_COSTS[c] for c in unlocked)
        attempts = 0
        while rem_budget >= min_cost and attempts < 1000:
            attempts += 1
            cand = int(np.random.choice(4, p=probs))
            if cand in unlocked and CLASS_COSTS[cand] <= rem_budget:
                army.append(cand)
                rem_budget -= CLASS_COSTS[cand]

        while rem_budget >= CLASS_COSTS[CLASS_GRUNT]:
            army.append(CLASS_GRUNT)
            rem_budget -= CLASS_COSTS[CLASS_GRUNT]

        return army

    def get_class_probabilities(self) -> np.ndarray:
        unlocked = self.get_unlocked_classes()
        masked_w = self.composition_weights.clone()
        for c in range(4):
            if c not in unlocked:
                masked_w[c] = -1e9
        probs = torch.softmax(masked_w, dim=0).cpu().numpy()
        return probs

    def learn_from_roi(self, class_roi_stats: dict[int, float], won_battle: bool):
        alpha = 0.20
        for c, roi in class_roi_stats.items():
            self.class_roi_ema[c] = (1.0 - alpha) * self.class_roi_ema[c] + alpha * roi

        ema_tensor = torch.tensor(self.class_roi_ema, dtype=torch.float32, device=self.device)
        if ema_tensor.max() > ema_tensor.min() + 1e-4:
            norm_roi = (ema_tensor - ema_tensor.mean()) / (ema_tensor.std() + 1e-5)
            lr = 0.15 if won_battle else 0.08
            self.composition_weights += lr * norm_roi
            self.composition_weights.clamp_(-3.0, 3.0)

        if self.active_techs:
            tech_reward = 0.18 if won_battle else -0.09
            for tid in self.active_techs:
                self.tech_weights[tid] += tech_reward
            self.tech_weights.clamp_(-2.5, 3.0)


class NeuroEvolutionPool:
    """
    Třídně izolovaný hierarchický genofond (14 -> 64 -> 64 -> 7).
    4 frakce x 4 třídy x 50 slotů = 800 slotů v GPU paměti.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.dtype = torch.float32

        self.N_IN = SENSOR_COUNT    # 14
        self.N_H1 = 64
        self.N_H2 = 64
        self.N_OUT = ACTION_COUNT   # 7
        self.N_FACTIONS = 4
        self.N_CLASSES = 4
        self.SLOTS_PER_CLASS = 50
        self.SLOTS_PER_FACTION = self.N_CLASSES * self.SLOTS_PER_CLASS  # 200
        self.MAX_UNITS = self.N_FACTIONS * self.SLOTS_PER_FACTION        # 800

        self.W1 = torch.zeros((self.MAX_UNITS, self.N_IN, self.N_H1), dtype=self.dtype, device=device)
        self.b1 = torch.zeros((self.MAX_UNITS, 1, self.N_H1), dtype=self.dtype, device=device)
        self.W2 = torch.zeros((self.MAX_UNITS, self.N_H1, self.N_H2), dtype=self.dtype, device=device)
        self.b2 = torch.zeros((self.MAX_UNITS, 1, self.N_H2), dtype=self.dtype, device=device)
        self.W3 = torch.zeros((self.MAX_UNITS, self.N_H2, self.N_OUT), dtype=self.dtype, device=device)
        self.b3 = torch.zeros((self.MAX_UNITS, 1, self.N_OUT), dtype=self.dtype, device=device)

        self.generals = [FactionGeneralBrain(f, device) for f in range(4)]

        self.epoch_number = 1
        self.history_best_fitness = [deque(maxlen=100) for _ in range(4)]
        self.history_avg_fitness = [deque(maxlen=100) for _ in range(4)]
        self.stagnation_counter = [0, 0, 0, 0]

        self._init_all_banks()

    def _init_all_banks(self):
        """Inicializuje váhy s vrozenými reflexy pro všech 7 akcí."""
        self.W1.normal_(0.0, 0.28)
        self.b1.zero_()
        self.W2.normal_(0.0, 0.25)
        self.b2.zero_()
        self.W3.normal_(0.0, 0.22)
        self.b3.zero_()

        # H1 percepce (vrozené synaptické dráhy):
        self.W1[:, 0, 0] = 0.95   # Cíl fwd -> H1[0]
        self.W1[:, 1, 1] = 0.95   # Cíl side -> H1[1]
        self.W1[:, 2, 2] = 1.20   # Cíl blízko -> H1[2]
        self.W1[:, 5, 3] = 0.80   # Povel fwd -> H1[3]
        self.W1[:, 6, 4] = 0.80   # Povel side -> H1[4]
        self.W1[:, 4, 5] = -0.85  # Palba zprava -> H1[5] (úhyb doleva)
        self.W1[:, 3, 6] = -0.75  # Palba zepředu -> H1[6] (zpátečka)
        self.W1[:, 10, 7] = 0.90  # Kryt fwd -> H1[7]
        self.W1[:, 11, 8] = 0.90  # Kryt side -> H1[8]
        self.W1[:, 7, 7] = 0.70   # Potlačení -> kryt H1[7]
        self.W1[:, 9, 9] = -0.90  # Nízké HP -> H1[9] (Danger/Survival reflex)
        self.W1[:, 12, 10] = 0.80 # Koheze & Ping čety -> H1[10]

        # H2 integrace:
        for i in range(min(self.N_H1, self.N_H2)):
            self.W2[:, i, i] = 1.05

        # H2 -> 7 Akcí:
        self.W3[:, 0, 0] = 0.85   # H2[0] -> Surge vpřed
        self.W3[:, 1, 1] = 0.75   # H2[1] -> Sway strafe
        self.W3[:, 1, 2] = 0.95   # H2[1] -> Yaw natočení na cíl
        self.W3[:, 2, 3] = 1.40   # H2[2] -> Trigger střelba
        self.W3[:, 3, 0] = 0.70   # H2[3] -> Surge dle povelu
        self.W3[:, 4, 1] = 0.70   # H2[4] -> Sway dle povelu
        self.W3[:, 5, 1] = 0.85   # H2[5] -> Sway úhyb před palbou
        self.W3[:, 6, 0] = -0.80  # H2[6] -> Surge couvání od palby
        self.W3[:, 7, 0] = 0.85   # H2[7] -> Surge ke krytu
        self.W3[:, 8, 1] = 0.85   # H2[8] -> Sway ke krytu

        # Stance, Ability a Ping:
        self.W3[:, 7, 4] = -0.80  # V krytu -> Stance: -1.0 (Zakleknutí / Zalehnutí za překážku)
        self.W3[:, 0, 4] = 0.65   # Volný postup k cíli -> Stance: +1.0 (Sprint přes pole)
        self.W3[:, 9, 5] = 1.30   # Nízké HP / nouze -> Ability: >0.60 (Aktivace schopnosti)
        self.W3[:, 2, 6] = 0.85   # Kontakt s nepřítelem na blízko -> Ping: 1.0 (Upozornit četu)

    def get_class_slots_for_army(self, faction_id: int, army_classes: list[int]) -> torch.Tensor:
        class_counters = [0, 0, 0, 0]
        slots = []
        for c in army_classes:
            idx = class_counters[c] % self.SLOTS_PER_CLASS
            class_counters[c] += 1
            global_slot = (faction_id * self.N_CLASSES + c) * self.SLOTS_PER_CLASS + idx
            slots.append(global_slot)
        return torch.tensor(slots, dtype=torch.long, device=self.device)

    def init_population(self, n_units: int, classes: torch.Tensor):
        self._init_all_banks()
        self.epoch_number = 1
        self.history_best_fitness = [deque(maxlen=100) for _ in range(4)]
        self.history_avg_fitness = [deque(maxlen=100) for _ in range(4)]

    def forward(self, brain_slots: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
        return tactical_brain_forward(
            sensors,
            self.W1[brain_slots], self.b1[brain_slots],
            self.W2[brain_slots], self.b2[brain_slots],
            self.W3[brain_slots], self.b3[brain_slots],
        )

    def inspect_decision_flow(self, brain_slot: int, sensor_vec: torch.Tensor) -> dict:
        with torch.no_grad():
            x_3d = sensor_vec.view(1, 1, self.N_IN)
            w1 = self.W1[brain_slot:brain_slot+1]
            b1 = self.b1[brain_slot:brain_slot+1]
            w2 = self.W2[brain_slot:brain_slot+1]
            b2 = self.b2[brain_slot:brain_slot+1]
            w3 = self.W3[brain_slot:brain_slot+1]
            b3 = self.b3[brain_slot:brain_slot+1]

            h1 = torch.tanh(torch.bmm(x_3d, w1) + b1)
            h2 = torch.tanh(torch.bmm(h1, w2) + b2)
            raw_out = torch.bmm(h2, w3) + b3

            surge_sway_yaw = torch.tanh(raw_out[:, :, 0:3])
            trigger = torch.sigmoid(raw_out[:, :, 3:4])
            stance = torch.tanh(raw_out[:, :, 4:5])
            ability = torch.sigmoid(raw_out[:, :, 5:6])
            ping = torch.tanh(raw_out[:, :, 6:7])
            out = torch.cat([surge_sway_yaw, trigger, stance, ability, ping], dim=2).squeeze().cpu().numpy()

            h1_np = h1.squeeze().cpu().numpy()
            h2_np = h2.squeeze().cpu().numpy()
            s_np = sensor_vec.cpu().numpy()

            sensors_summary = [(SENSOR_NAMES[i], float(s_np[i])) for i in range(self.N_IN)]
            actions_summary = [(ACTION_NAMES[i], float(out[i])) for i in range(self.N_OUT)]

            return {
                "sensors": sensors_summary,
                "actions": actions_summary,
                "hidden_h1": [float(v) for v in h1_np],
                "hidden_h2": [float(v) for v in h2_np],
                "hidden": [float(v) for v in np.concatenate([h1_np, h2_np])],
            }

    def evolve_epoch(
        self,
        brain_slots: torch.Tensor,
        teams: torch.Tensor,
        classes: torch.Tensor,
        fitness: torch.Tensor,
        n_factions: int,
        winner_team: int = -1,
        mutation_rate: float = 0.08,
    ):
        self.epoch_number += 1

        for f in range(n_factions):
            f_mask = teams == f
            f_indices = torch.where(f_mask)[0]
            if f_indices.numel() < 4:
                continue

            f_fit = fitness[f_indices]
            f_classes = classes[f_indices]
            f_slots = brain_slots[f_indices]

            class_roi_stats = {}
            for c in range(4):
                c_mask = f_classes == c
                if c_mask.any():
                    avg_c_fit = float(torch.mean(f_fit[c_mask]).item())
                    class_roi_stats[c] = avg_c_fit / float(CLASS_COSTS[c])

            self.generals[f].update_level()
            self.generals[f].learn_from_roi(class_roi_stats, won_battle=(f == winner_team))

            cur_best = float(torch.max(f_fit).item())
            cur_avg = float(torch.mean(f_fit).item())
            prev_best = self.history_best_fitness[f][-1] if self.history_best_fitness[f] else 0.0

            if cur_best <= prev_best * 1.015:
                self.stagnation_counter[f] += 1
            else:
                self.stagnation_counter[f] = 0

            self.history_best_fitness[f].append(cur_best)
            self.history_avg_fitness[f].append(cur_avg)

            stag_epochs = self.stagnation_counter[f]
            is_stagnating = stag_epochs >= 3

            for c in range(4):
                c_unit_mask = f_classes == c
                c_units = torch.where(c_unit_mask)[0]
                c_count = c_units.numel()

                base_slot = (f * self.N_CLASSES + c) * self.SLOTS_PER_CLASS
                all_class_slots = torch.arange(base_slot, base_slot + self.SLOTS_PER_CLASS, dtype=torch.long, device=self.device)

                if c_count < 2:
                    self.W1[all_class_slots] += torch.randn_like(self.W1[all_class_slots]) * (mutation_rate * 0.4)
                    self.W2[all_class_slots] += torch.randn_like(self.W2[all_class_slots]) * (mutation_rate * 0.4)
                    self.W3[all_class_slots] += torch.randn_like(self.W3[all_class_slots]) * (mutation_rate * 0.4)
                    continue

                c_fit = f_fit[c_units]
                c_active_slots = f_slots[c_units]

                _, sorted_rel = torch.sort(c_fit, descending=True)
                sorted_slots = c_active_slots[sorted_rel]

                n_elite = max(2, min(5, c_count // 3))
                elite_slots = sorted_slots[:n_elite]
                parent_pool = sorted_slots[:max(n_elite + 1, c_count // 2)]

                self.W1[base_slot:base_slot + n_elite] = self.W1[elite_slots].clone()
                self.b1[base_slot:base_slot + n_elite] = self.b1[elite_slots].clone()
                self.W2[base_slot:base_slot + n_elite] = self.W2[elite_slots].clone()
                self.b2[base_slot:base_slot + n_elite] = self.b2[elite_slots].clone()
                self.W3[base_slot:base_slot + n_elite] = self.W3[elite_slots].clone()
                self.b3[base_slot:base_slot + n_elite] = self.b3[elite_slots].clone()

                target_slots = all_class_slots[n_elite:]
                n_children = target_slots.numel()
                n_parents = parent_pool.numel()

                p1_idx = torch.randint(0, n_parents, (n_children,), device=self.device)
                p2_idx = torch.randint(0, n_parents, (n_children,), device=self.device)

                parent_W1 = self.W1[parent_pool]
                parent_b1 = self.b1[parent_pool]
                parent_W2 = self.W2[parent_pool]
                parent_b2 = self.b2[parent_pool]
                parent_W3 = self.W3[parent_pool]
                parent_b3 = self.b3[parent_pool]

                mask1 = torch.rand((n_children, self.N_IN, self.N_H1), device=self.device) > 0.5
                child_W1 = torch.where(mask1, parent_W1[p1_idx], parent_W1[p2_idx])
                child_b1 = 0.5 * (parent_b1[p1_idx] + parent_b1[p2_idx])

                mask2 = torch.rand((n_children, self.N_H1, self.N_H2), device=self.device) > 0.5
                child_W2 = torch.where(mask2, parent_W2[p1_idx], parent_W2[p2_idx])
                child_b2 = 0.5 * (parent_b2[p1_idx] + parent_b2[p2_idx])

                mask3 = torch.rand((n_children, self.N_H2, self.N_OUT), device=self.device) > 0.5
                child_W3 = torch.where(mask3, parent_W3[p1_idx], parent_W3[p2_idx])
                child_b3 = 0.5 * (parent_b3[p1_idx] + parent_b3[p2_idx])

                m = mutation_rate * (0.85 if f == winner_team else 1.20)
                if is_stagnating:
                    temp_mult = min(3.2, 1.8 + 0.3 * stag_epochs)
                    m *= temp_mult

                    shock_mask1 = torch.rand_like(child_W1) < 0.15
                    child_W1 = torch.where(shock_mask1, torch.randn_like(child_W1) * 0.40, child_W1)
                    shock_mask2 = torch.rand_like(child_W2) < 0.15
                    child_W2 = torch.where(shock_mask2, torch.randn_like(child_W2) * 0.35, child_W2)

                child_W1 += torch.randn_like(child_W1) * m
                child_b1 += torch.randn_like(child_b1) * (m * 0.5)
                child_W2 += torch.randn_like(child_W2) * m
                child_b2 += torch.randn_like(child_b2) * (m * 0.5)
                child_W3 += torch.randn_like(child_W3) * m
                child_b3 += torch.randn_like(child_b3) * (m * 0.5)

                child_W1.mul_(0.988)
                child_W2.mul_(0.988)
                child_W3.mul_(0.988)

                self.W1[target_slots] = child_W1
                self.b1[target_slots] = child_b1
                self.W2[target_slots] = child_W2
                self.b2[target_slots] = child_b2
                self.W3[target_slots] = child_W3
                self.b3[target_slots] = child_b3
