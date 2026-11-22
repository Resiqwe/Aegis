r"""
tests/test_golden_master.py - Záchranná síť (Golden Master) pro refaktoring engine_core.

Ověřuje, že simulace s pevným seedem produkuje identické výsledky:
- Po 1. ticku s přísnou tolerancí (atol=1e-6, rtol=1e-6) pro okamžité odhalení matematických odchylek.
- Po 300. ticku s tolerancí (atol=1e-4, rtol=1e-4) pro zachycení kumulativních změn.

Pro přegenerování referenčního otisku:
    $env:GENERATE_GOLDEN="1"; .\.venv\Scripts\python.exe -m pytest tests/test_golden_master.py
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine_core import RealisticBattleEngine  # noqa: E402

GOLDEN_PATH = TEST_DIR / "golden" / "golden_state.pt"
SEED = 424242
N_UNITS = 64
N_FACTIONS = 4
DT = 1.0 / 60.0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_state(engine: RealisticBattleEngine) -> dict[str, torch.Tensor]:
    return {
        "pos": engine.pos.detach().cpu().clone(),
        "vel": engine.vel.detach().cpu().clone(),
        "heading": engine.heading.detach().cpu().clone(),
        "hp": engine.hp.detach().cpu().clone(),
        "ammo_in_mag": engine.ammo_in_mag.detach().cpu().clone(),
        "suppression": engine.suppression.detach().cpu().clone(),
        "fitness": engine.fitness.detach().cpu().clone(),
        "damage_dealt": engine.damage_dealt.detach().cpu().clone(),
        "kills_per_faction": engine.kills_per_faction.detach().cpu().clone(),
        "unit_kills": engine.unit_kills.detach().cpu().clone(),
        "node_control_points": engine.node_control_points.detach().cpu().clone(),
    }


def _run_simulation() -> dict[str, dict[str, torch.Tensor]]:
    _seed_everything(SEED)
    engine = RealisticBattleEngine()
    engine.visual_mode = False
    engine.configure(n_units=N_UNITS, n_factions=N_FACTIONS)
    engine.switch_scenario(0)

    engine.step(DT)
    step1 = _capture_state(engine)

    for _ in range(299):
        engine.step(DT)
    step300 = _capture_state(engine)

    return {"step1": step1, "step300": step300}


def _assert_states_close(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    atol: float,
    rtol: float,
    label: str,
) -> None:
    for key, exp_t in expected.items():
        act_t = actual[key]
        assert act_t.shape == exp_t.shape, f"[{label}] Shape mismatch for {key}: {act_t.shape} vs {exp_t.shape}"
        if exp_t.is_floating_point():
            max_diff = float((act_t - exp_t).abs().max().item()) if exp_t.numel() > 0 else 0.0
            assert torch.allclose(act_t, exp_t, atol=atol, rtol=rtol), (
                f"[{label}] Tensor '{key}' diverged! max_abs_diff={max_diff:.3e} (atol={atol}, rtol={rtol})"
            )
        else:
            assert torch.equal(act_t, exp_t), f"[{label}] Integer tensor '{key}' mismatch!"


def test_golden_master_determinism():
    results = _run_simulation()

    if os.environ.get("GENERATE_GOLDEN") == "1" or not GOLDEN_PATH.exists():
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(results, GOLDEN_PATH)
        pytest.skip(f"Vygenerován nový golden master otisk: {GOLDEN_PATH}")

    expected = torch.load(GOLDEN_PATH, map_location="cpu", weights_only=True)
    _assert_states_close(results["step1"], expected["step1"], atol=1e-6, rtol=1e-6, label="step1")
    _assert_states_close(results["step300"], expected["step300"], atol=1e-4, rtol=1e-4, label="step300")
