"""
engine/tick_context.py - Sdílený datový kontext jednoho simulačního ticku.

Uchovává mezivýsledky (GPU tenzory) předávané mezi bezstavovými systémy
(sensors -> locomotion -> combat -> commit -> objectives) bez kopírování
a bez jakýchkoli GPU->CPU synchronizací.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TickContext:
    dt: float
    N: int
    alive: torch.Tensor
    alive_idx: torch.Tensor
    K: int

    # Terén / kryt
    wall_force: Optional[torch.Tensor] = None

    # Senzory a cílení (vyplňuje sensors_system)
    dir_enemy: Optional[torch.Tensor] = None
    dist_enemy: Optional[torch.Tensor] = None
    target_enemy: Optional[torch.Tensor] = None
    medic_patient_dir: Optional[torch.Tensor] = None
    medic_has_patient: Optional[torch.Tensor] = None

    # Pár-na-pár matice pro živé jednotky (K > 1)
    p_alive: Optional[torch.Tensor] = None
    t_alive: Optional[torch.Tensor] = None
    dist_mat: Optional[torch.Tensor] = None
    diff_all: Optional[torch.Tensor] = None
    same_team: Optional[torch.Tensor] = None

    # Orientace a výstup neuronové sítě
    cos_th: Optional[torch.Tensor] = None
    sin_th: Optional[torch.Tensor] = None
    brain_out: Optional[torch.Tensor] = None

    # Locomotion -> Combat předávka (bez duplicitních atan2)
    has_target: Optional[torch.Tensor] = None
    tar_diff: Optional[torch.Tensor] = None
    brain_trigger: Optional[torch.Tensor] = None
