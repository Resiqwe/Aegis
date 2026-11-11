"""
staging_buffer.py - SimulationStagingBuffer pro deterministickou simulaci.

Všechny subsystémy (balistika, léčení, schopnosti) zapisují výhradně delty do tohoto bufferu.
Engine provede jednorázový atomický commit na konci fáze exekuce.
"""

import torch


class SimulationStagingBuffer:
    """
    Zero-Trust Staging Buffer pro deterministickou simulaci bez rozptýlených zápisů.
    """
    def __init__(self, num_units: int, device: torch.device):
        self.num_units = num_units
        self.device = device

        self.delta_hp = torch.zeros((num_units,), dtype=torch.float32, device=device)
        self.delta_vel = torch.zeros((num_units, 2), dtype=torch.float32, device=device)
        self.delta_suppression = torch.zeros((num_units,), dtype=torch.float32, device=device)
        self.stagger_duration = torch.zeros((num_units,), dtype=torch.float32, device=device)
        self.damage_taken = torch.zeros((num_units,), dtype=torch.float32, device=device)
        self.out_of_cover_penalty = torch.zeros((num_units,), dtype=torch.float32, device=device)
        
        # Sledování posledního střelce/útočníka pro korektní atribuci killů
        self.last_attacker_id = torch.full((num_units,), -1, dtype=torch.long, device=device)

    def reset(self):
        self.delta_hp.zero_()
        self.delta_vel.zero_()
        self.delta_suppression.zero_()
        self.stagger_duration.zero_()
        self.damage_taken.zero_()
        self.out_of_cover_penalty.zero_()
        self.last_attacker_id.fill_(-1)
