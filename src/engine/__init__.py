"""
engine/__init__.py - Hlavní vstupní bod balíčku Aegis Engine 2.0.
"""

from engine.state_bridge import LockFreeStateBridge, InterpolatedRenderSnapshot
from engine.async_runner import AsyncSimulationWorker
from engine.staging_buffer import SimulationStagingBuffer
from engine.spatial_system import SpatialSystem, check_line_of_sight
from engine.ballistics_system import VectorizedBallisticsSystem
from engine.combat_system import CombatSystem
from engine.objectives_system import ObjectivesSystem

__all__ = [
    "LockFreeStateBridge",
    "InterpolatedRenderSnapshot",
    "AsyncSimulationWorker",
    "SimulationStagingBuffer",
    "SpatialSystem",
    "check_line_of_sight",
    "VectorizedBallisticsSystem",
    "CombatSystem",
    "ObjectivesSystem",
]
