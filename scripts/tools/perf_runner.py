"""
scripts/tools/perf_runner.py - Sdílený engine runner pro GPU/CPU profiling, VRAM audit,
deterministický benchmark a kontrolu datových kontraktů (NaN/Inf/shape/dtype/device).
"""

from __future__ import annotations

import inspect
import json
import math
import sys
import textwrap
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _ensure_import_paths(root: Path) -> None:
    src_dir = str(root / "src")
    root_str = str(root)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_engine(root: Path, n_units: int, seed: int):
    _ensure_import_paths(root)
    import engine_core

    _seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    engine = engine_core.RealisticBattleEngine(2400, 1350, device)
    engine.configure(n_units, 4)
    engine.visual_mode = False
    return engine, engine_core, device


def run_profile(
    root: Path,
    n_units: int = 800,
    warmup: int = 30,
    ticks: int = 200,
    seed: int = 42,
    compact: bool = False,
    as_json: bool = False,
) -> int:
    engine, engine_core, device = _create_engine(root, n_units, seed)
    is_cuda = device.type == "cuda"

    # Warm-up bez wrapperů (zahřátí CUDA kontextu a caching alokátoru)
    for _ in range(warmup):
        engine.step(1.0 / 60.0)
    if is_cuda:
        torch.cuda.synchronize()

    # Sledování alokací a VRAM po warm-upu
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device)
        stats_before = torch.cuda.memory_stats(device)
        alloc_before = stats_before.get("allocation.all.allocated", 0)
    else:
        mem_before = 0
        alloc_before = 0

    cpu_totals: dict[str, float] = defaultdict(float)
    gpu_totals: dict[str, float] = defaultdict(float)
    call_counts: dict[str, int] = defaultdict(int)
    tick_events: list[tuple[str, Any, Any]] = []

    def make_wrapper(name: str, orig_fn):
        def wrapped(*args, **kwargs):
            if is_cuda:
                s_evt = torch.cuda.Event(enable_timing=True)
                e_evt = torch.cuda.Event(enable_timing=True)
                s_evt.record()
            else:
                s_evt = e_evt = None
            t0 = time.perf_counter()
            res = orig_fn(*args, **kwargs)
            t1 = time.perf_counter()
            if is_cuda:
                e_evt.record()
                tick_events.append((name, s_evt, e_evt))
            cpu_totals[name] += (t1 - t0) * 1000.0
            call_counts[name] += 1
            return res
        return wrapped

    # Neinvazivní instrumentace 3 vnitřních sekcí _step_internal bez úpravy engine_core.py
    class _SectionTimer:
        def __init__(self, sec_name: str):
            self.sec_name = sec_name
            self.s_evt = None
            self.t0 = 0.0

        def __enter__(self):
            if is_cuda:
                self.s_evt = torch.cuda.Event(enable_timing=True)
                self.s_evt.record()
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            t1 = time.perf_counter()
            if is_cuda:
                e_evt = torch.cuda.Event(enable_timing=True)
                e_evt.record()
                tick_events.append((self.sec_name, self.s_evt, e_evt))
            cpu_totals[self.sec_name] += (t1 - self.t0) * 1000.0
            call_counts[self.sec_name] += 1
            return False

    def _build_instrumented_step(orig_method):
        src = textwrap.dedent(inspect.getsource(orig_method))
        lines = src.splitlines()
        out_lines = []
        active_sec = None
        for line in lines:
            stripped = line.strip()
            # Vstup do 1. sekce: sensors_and_ego
            if stripped == "self.staging_buf.reset()":
                out_lines.append("    _sec_ctx = _aegis_sec('sensors_and_ego'); _sec_ctx.__enter__()")
                active_sec = "sensors_and_ego"
            # Přechod na 2. sekci: movement_and_steering
            elif stripped.startswith("squad_force = torch.zeros((N, 2)"):
                if active_sec:
                    out_lines.append("    _sec_ctx.__exit__(None, None, None)")
                out_lines.append("    _sec_ctx = _aegis_sec('movement_and_steering'); _sec_ctx.__enter__()")
                active_sec = "movement_and_steering"
            # Přechod na 3. sekci: targeting_and_weapons
            elif stripped.startswith("utility = torch.zeros((N, 6)"):
                if active_sec:
                    out_lines.append("    _sec_ctx.__exit__(None, None, None)")
                out_lines.append("    _sec_ctx = _aegis_sec('targeting_and_weapons'); _sec_ctx.__enter__()")
                active_sec = "targeting_and_weapons"
            # Konec 3. sekce před _atomic_commit_pipeline
            elif stripped == "self._atomic_commit_pipeline(dt)":
                if active_sec:
                    out_lines.append("    _sec_ctx.__exit__(None, None, None)")
                    active_sec = None
            out_lines.append(line)
        ns = dict(engine_core.__dict__)
        ns["_aegis_sec"] = _SectionTimer
        exec("\n".join(out_lines), ns)
        return types.MethodType(ns["_step_internal"], engine)

    # Neinvazivní monkey-patch klíčových subsystémů + 3 sekcí _step_internal
    orig_step_internal = engine._step_internal
    orig_spatial_col = engine.spatial.resolve_obstacle_collisions
    orig_los = engine_core.check_line_of_sight
    orig_brains = engine.brains.forward
    orig_combat = engine.combat.process_abilities
    orig_ballistics = engine.ballistics.resolve_ballistic_hits
    orig_gates = engine.objectives.update_gates
    orig_status = engine.objectives.check_battle_status
    orig_commit = engine._atomic_commit_pipeline

    engine._step_internal = _build_instrumented_step(engine_core.RealisticBattleEngine._step_internal)
    engine.spatial.resolve_obstacle_collisions = make_wrapper("spatial.resolve_obstacle_collisions", orig_spatial_col)
    engine_core.check_line_of_sight = make_wrapper("spatial.check_line_of_sight", orig_los)
    engine.brains.forward = make_wrapper("brains.forward", orig_brains)
    engine.combat.process_abilities = make_wrapper("combat.process_abilities", orig_combat)
    engine.ballistics.resolve_ballistic_hits = make_wrapper("ballistics.resolve_ballistic_hits", orig_ballistics)
    engine.objectives.update_gates = make_wrapper("objectives.update_gates", orig_gates)
    engine.objectives.check_battle_status = make_wrapper("objectives.check_battle_status", orig_status)
    engine._atomic_commit_pipeline = make_wrapper("_atomic_commit_pipeline", orig_commit)

    total_wall_ms = 0.0
    total_cpu_step_ms = 0.0
    try:
        for _ in range(ticks):
            tick_events.clear()
            t_w0 = time.perf_counter()
            engine.step(1.0 / 60.0)
            t_cpu1 = time.perf_counter()
            if is_cuda:
                torch.cuda.synchronize()
            t_w1 = time.perf_counter()

            total_cpu_step_ms += (t_cpu1 - t_w0) * 1000.0
            total_wall_ms += (t_w1 - t_w0) * 1000.0

            if is_cuda:
                for name, s_evt, e_evt in tick_events:
                    gpu_totals[name] += float(s_evt.elapsed_time(e_evt))
    finally:
        engine._step_internal = orig_step_internal
        engine.spatial.resolve_obstacle_collisions = orig_spatial_col
        engine_core.check_line_of_sight = orig_los
        engine.brains.forward = orig_brains
        engine.combat.process_abilities = orig_combat
        engine.ballistics.resolve_ballistic_hits = orig_ballistics
        engine.objectives.update_gates = orig_gates
        engine.objectives.check_battle_status = orig_status
        engine._atomic_commit_pipeline = orig_commit

    if is_cuda:
        stats_after = torch.cuda.memory_stats(device)
        alloc_after = stats_after.get("allocation.all.allocated", 0)
        alloc_calls_per_tick = round((alloc_after - alloc_before) / max(1, ticks), 1)
        peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
        vram_delta_mb = round((torch.cuda.max_memory_allocated(device) - mem_before) / (1024 * 1024), 2)
        dev_name = torch.cuda.get_device_name(0)
    else:
        alloc_calls_per_tick = 0.0
        peak_vram_mb = 0.0
        vram_delta_mb = 0.0
        dev_name = "CPU"

    avg_wall_ms = total_wall_ms / max(1, ticks)
    avg_cpu_step_ms = total_cpu_step_ms / max(1, ticks)
    top_level_keys = (
        "sensors_and_ego",
        "movement_and_steering",
        "targeting_and_weapons",
        "_atomic_commit_pipeline",
        "objectives.update_gates",
        "objectives.check_battle_status",
    )
    top_level_cpu_sum = sum(cpu_totals.get(k, 0.0) / max(1, ticks) for k in top_level_keys)
    uninstrumented_cpu_ms = max(0.0, avg_cpu_step_ms - top_level_cpu_sum)

    section_names = {"sensors_and_ego", "movement_and_steering", "targeting_and_weapons"}
    rows = [
        {
            "name": "uninstrumented_remainder (init/nodes/particles)",
            "cpu_dispatch_ms": round(uninstrumented_cpu_ms, 3),
            "gpu_kernel_ms": None,
            "calls": ticks,
        }
    ]
    for name in sorted(cpu_totals.keys(), key=lambda k: cpu_totals[k], reverse=True):
        label = f"[section] {name}" if name in section_names else name
        rows.append({
            "name": label,
            "cpu_dispatch_ms": round(cpu_totals[name] / max(1, ticks), 3),
            "gpu_kernel_ms": round(gpu_totals[name] / max(1, ticks), 3) if is_cuda else None,
            "calls": call_counts[name],
        })
    rows.sort(key=lambda r: r["cpu_dispatch_ms"], reverse=True)

    result = {
        "n_units": n_units,
        "warmup": warmup,
        "ticks": ticks,
        "device": f"{device} ({dev_name})",
        "avg_tick_wall_ms": round(avg_wall_ms, 3),
        "avg_tick_cpu_dispatch_ms": round(avg_cpu_step_ms, 3),
        "tps": round(1000.0 / max(1e-6, avg_wall_ms), 1),
        "peak_vram_mb": peak_vram_mb,
        "vram_delta_mb": vram_delta_mb,
        "alloc_calls_per_tick": alloc_calls_per_tick,
        "subsystems": rows,
    }

    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"[AEGIS PROFILE] N={n_units} warmup={warmup} ticks={ticks} device={result['device']}")
        print(
            f"AVG_TICK: wall={result['avg_tick_wall_ms']:.2f}ms (TPS={result['tps']}) | "
            f"VRAM: peak={peak_vram_mb:.1f}MB (delta=+{vram_delta_mb:.1f}MB) | alloc_calls_per_tick={alloc_calls_per_tick}"
        )
        print("SUBSYSTEMS (avg per tick: cpu_dispatch_ms / gpu_kernel_ms):")
        for idx, r in enumerate(rows[: (12 if compact else 20)], 1):
            g_str = f"{r['gpu_kernel_ms']:6.3f}ms" if r["gpu_kernel_ms"] is not None else "     --"
            print(f"  {idx}. {r['name']:<50} : {r['cpu_dispatch_ms']:6.3f}ms / {g_str}")
    return 0


def _check_tensor_contracts(engine, device: torch.device) -> list[str]:
    errors: list[str] = []
    N = engine.N
    checks = [
        ("pos", engine.pos, (N, 2), torch.float32),
        ("vel", engine.vel, (N, 2), torch.float32),
        ("hp", engine.hp, (N,), torch.float32),
        ("teams", engine.teams, (N,), None),
        ("classes", engine.classes, (N,), None),
        ("fitness", engine.fitness, (N,), torch.float32),
        ("W1", engine.brains.W1, (engine.brains.MAX_UNITS, engine.brains.N_IN, engine.brains.N_H1), torch.float32),
        ("W2", engine.brains.W2, (engine.brains.MAX_UNITS, engine.brains.N_H1, engine.brains.N_H2), torch.float32),
        ("W3", engine.brains.W3, (engine.brains.MAX_UNITS, engine.brains.N_H2, engine.brains.N_OUT), torch.float32),
    ]
    for name, t, exp_shape, exp_dtype in checks:
        if tuple(t.shape) != exp_shape:
            errors.append(f"{name}: shape={tuple(t.shape)} != {exp_shape}")
        if exp_dtype is not None and t.dtype != exp_dtype:
            errors.append(f"{name}: dtype={t.dtype} != {exp_dtype}")
        if t.device.type != device.type:
            errors.append(f"{name}: device={t.device} != {device}")
        if t.is_floating_point():
            if torch.isnan(t).any().item() or torch.isinf(t).any().item():
                errors.append(f"{name}: contains NaN/Inf")
    if engine.nodes_gpu.numel() > 0:
        if torch.isnan(engine.nodes_gpu).any().item() or torch.isinf(engine.nodes_gpu).any().item():
            errors.append("nodes_gpu: contains NaN/Inf")
    return errors


def _collect_bench_metrics(engine) -> dict[str, Any]:
    return {
        "tick_counter": int(engine.tick_counter),
        "alive_count": int((engine.hp > 0.0).sum().item()),
        "kills_per_faction": [int(x) for x in engine.kills_per_faction.detach().cpu().tolist()],
        "node_owners": [int(x) for x in engine.nodes_gpu[:, 3].detach().cpu().tolist()] if engine.nodes_gpu.numel() > 0 else [],
        "node_control_points": [float(x) for x in engine.node_control_points.detach().cpu().tolist()],
        "pos_sum": float(engine.pos.sum().item()),
        "hp_sum": float(engine.hp.sum().item()),
        "fitness_sum": float(engine.fitness.sum().item()),
        "fitness_std": float(engine.fitness.std().item()),
    }


def _compare_metrics(base: dict[str, Any], curr: dict[str, Any]) -> tuple[str, list[str]]:
    notes: list[str] = []
    exact = True
    within_tol = True

    for key in ("tick_counter", "alive_count"):
        diff = abs(curr[key] - base[key])
        if diff > 0:
            exact = False
            if diff > 2:
                within_tol = False
                notes.append(f"{key}: {base[key]} -> {curr[key]} (diff={diff})")

    for key in ("kills_per_faction", "node_owners"):
        if curr[key] != base[key]:
            exact = False
            max_d = max((abs(a - b) for a, b in zip(base[key], curr[key])), default=99)
            if max_d > 2 or len(curr[key]) != len(base[key]):
                within_tol = False
                notes.append(f"{key}: {base[key]} -> {curr[key]}")

    for key in ("pos_sum", "hp_sum", "fitness_sum", "fitness_std"):
        if not math.isclose(curr[key], base[key], rel_tol=1e-4, abs_tol=1e-4):
            exact = False
            if not math.isclose(curr[key], base[key], rel_tol=1e-3, abs_tol=1e-2):
                within_tol = False
                notes.append(f"{key}: {base[key]:.4f} -> {curr[key]:.4f}")

    for b_pt, c_pt in zip(base["node_control_points"], curr["node_control_points"]):
        if not math.isclose(c_pt, b_pt, rel_tol=1e-4, abs_tol=1e-4):
            exact = False
            if not math.isclose(c_pt, b_pt, rel_tol=1e-3, abs_tol=1e-2):
                within_tol = False
                notes.append(f"node_control_points: {base['node_control_points']} -> {curr['node_control_points']}")
                break

    if exact:
        return "EXACT", notes
    if within_tol:
        return "WITHIN_TOL", notes
    return "REGRESSION", notes


def run_bench(
    root: Path,
    n_units: int = 800,
    ticks: int = 500,
    seed: int = 42,
    save_baseline: bool = False,
    compare: bool = False,
    as_json: bool = False,
) -> int:
    engine, _, device = _create_engine(root, n_units, seed)
    is_cuda = device.type == "cuda"

    # Warm-up 15 tiků
    for _ in range(15):
        engine.step(1.0 / 60.0)
    if is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(ticks):
        engine.step(1.0 / 60.0)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    metrics = _collect_bench_metrics(engine)
    contract_errors = _check_tensor_contracts(engine, device)

    # Ověření evoluce + kontraktu vah po 1 epoše
    engine.finish_operation_and_evolve()
    contract_errors.extend(_check_tensor_contracts(engine, device))

    payload = {
        "n_units": n_units,
        "ticks": ticks,
        "seed": seed,
        "device": str(device),
        "total_sec": round(elapsed, 4),
        "ms_per_tick": round((elapsed / max(1, ticks)) * 1000.0, 3),
        "tps": round(ticks / max(1e-6, elapsed), 1),
        "contract_errors": contract_errors,
        "metrics": metrics,
    }

    baseline_path = root / "scripts" / "baseline_bench.json"

    if save_baseline:
        baseline_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    verdict = None
    diff_notes: list[str] = []
    tps_delta_pct = None
    if compare:
        if not baseline_path.exists():
            print(f"[AEGIS BENCH] Baseline nenalezen ({baseline_path}). Nejprve spusť --save-baseline.")
            return 1
        base = json.loads(baseline_path.read_text(encoding="utf-8"))
        verdict, diff_notes = _compare_metrics(base["metrics"], metrics)
        if contract_errors:
            verdict = "REGRESSION"
        tps_delta_pct = round(((payload["tps"] - base["tps"]) / max(1e-6, base["tps"])) * 100.0, 1)
        payload["compare"] = {
            "baseline_tps": base["tps"],
            "tps_delta_pct": tps_delta_pct,
            "verdict": verdict,
            "notes": diff_notes,
        }

    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        status_c = "OK (0 NaN/Inf)" if not contract_errors else f"FAIL ({len(contract_errors)} errors)"
        print(
            f"[AEGIS BENCH] N={n_units} ticks={ticks} seed={seed} ({device}) | "
            f"total={payload['total_sec']:.3f}s | {payload['ms_per_tick']:.2f} ms/tick | TPS={payload['tps']} | CONTRACTS={status_c}"
        )
        if contract_errors:
            for e in contract_errors:
                print(f"  CONTRACT_ERROR: {e}")
        if save_baseline:
            print(f"  Baseline uložen -> {baseline_path}")
        if compare and verdict is not None:
            print(f"  COMPARE: TPS {payload['compare']['baseline_tps']} -> {payload['tps']} ({tps_delta_pct:+.1f}%) | LOGIC MATCH: {verdict}")
            for n in diff_notes:
                print(f"    DIFF: {n}")

    if contract_errors or verdict == "REGRESSION":
        return 1
    return 0
