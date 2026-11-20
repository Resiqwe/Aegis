"""
scripts/tools/snap_inspector.py - CLI inspektor uložených .pt snapshotů z telemetry/.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def _load_pt_cpu(path: Path) -> dict[str, Any]:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(str(path), map_location="cpu", weights_only=False)


def _tensor_health(t: torch.Tensor) -> dict[str, Any]:
    f = t.float()
    n_nan = int(torch.isnan(f).sum().item())
    n_inf = int(torch.isinf(f).sum().item())
    near_zero_pct = float((f.abs() < 1e-4).float().mean().item() * 100.0)
    return {
        "shape": list(t.shape),
        "min": round(float(f.min().item()), 4),
        "max": round(float(f.max().item()), 4),
        "std": round(float(f.std().item()), 4),
        "near_zero_pct": round(near_zero_pct, 2),
        "nan": n_nan,
        "inf": n_inf,
    }


def list_snapshots(root: Path, telemetry_dir: str = "telemetry", as_json: bool = False) -> int:
    tdir = root / telemetry_dir
    if not tdir.exists():
        print(f"[AEGIS SNAPS] Adresář '{tdir}' neexistuje.")
        return 0

    pt_files = sorted(tdir.glob("snapshot_*.pt"), reverse=True)
    rows = []
    for p in pt_files:
        size_mb = round(p.stat().st_size / (1024 * 1024), 2)
        mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        meta = {"file": p.name, "size_mb": size_mb, "mtime": mtime, "epoch": "?", "scenario": "?", "wins": "?"}
        try:
            data = _load_pt_cpu(p)
            meta["epoch"] = data.get("epoch", "?")
            meta["scenario"] = data.get("scenario_idx", "?")
            meta["wins"] = data.get("wins_per_faction", "?")
        except Exception as exc:
            meta["error"] = str(exc)
        rows.append(meta)

    if as_json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    if not rows:
        print(f"[AEGIS SNAPS] V '{telemetry_dir}/' nejsou žádné snapshoty (.pt).")
        return 0

    print(f"[AEGIS SNAPS] Nalezeno {len(rows)} snapshotů v '{telemetry_dir}/':")
    print(f"{'Soubor':<46} {'MB':>6} {'Ep':>4} {'Sc':>3} {'Výhry':<18} {'Datum'}")
    for r in rows:
        print(f"{r['file']:<46} {r['size_mb']:>6.2f} {str(r['epoch']):>4} {str(r['scenario']):>3} {str(r['wins']):<18} {r['mtime']}")
    return 0


def inspect_snapshot(path_str: str, as_json: bool = False) -> int:
    p = Path(path_str)
    if not p.exists():
        print(f"[AEGIS SNAPS] Soubor nenalezen: {p}")
        return 1

    data = _load_pt_cpu(p)
    is_full = "brains_W1_full" in data
    has_legacy = "brains_W1" in data

    weights_health = {}
    for k in ("brains_W1_full", "brains_W2_full", "brains_W3_full", "brains_W1", "brains_W2"):
        if k in data and isinstance(data[k], torch.Tensor):
            weights_health[k] = _tensor_health(data[k])

    generals = []
    xp_list = data.get("generals_xp", [])
    lvl_list = data.get("generals_level", [])
    roi_list = data.get("generals_roi_ema", [])
    for i in range(len(xp_list)):
        generals.append({
            "faction": i,
            "level": lvl_list[i] if i < len(lvl_list) else 0,
            "xp": round(float(xp_list[i]), 1),
            "roi_ema": [round(float(x), 2) for x in (roi_list[i] if i < len(roi_list) else [])],
        })

    info = {
        "file": p.name,
        "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
        "format": "full+legacy" if (is_full and has_legacy) else ("full" if is_full else "legacy"),
        "epoch": data.get("epoch"),
        "scenario_idx": data.get("scenario_idx"),
        "n_units": data.get("n_units"),
        "wins_per_faction": data.get("wins_per_faction"),
        "faction_doctrines": data.get("faction_doctrines"),
        "generals": generals,
        "weights_health": weights_health,
    }

    if as_json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    print(
        f"[AEGIS SNAP INSPECT] {info['file']} ({info['size_mb']} MB, format={info['format']})\n"
        f"  epoch={info['epoch']} | scenario={info['scenario_idx']} | N={info['n_units']} | wins={info['wins_per_faction']}"
    )
    print("  GENERALS:")
    for g in generals:
        print(f"    F{g['faction']}: lvl={g['level']} xp={g['xp']} roi_ema={g['roi_ema']}")
    print("  WEIGHTS HEALTH:")
    for k, h in weights_health.items():
        print(f"    {k:<15}: shape={h['shape']} min={h['min']} max={h['max']} std={h['std']} zero%={h['near_zero_pct']} nan={h['nan']} inf={h['inf']}")
    return 0


def diff_snapshots(path_a: str, path_b: str, as_json: bool = False) -> int:
    pa, pb = Path(path_a), Path(path_b)
    if not pa.exists() or not pb.exists():
        print(f"[AEGIS SNAPS] Jeden ze souborů neexistuje: {pa}, {pb}")
        return 1

    da, db = _load_pt_cpu(pa), _load_pt_cpu(pb)
    xp_a = da.get("generals_xp", [])
    xp_b = db.get("generals_xp", [])
    xp_delta = [round(float(xp_b[i]) - float(xp_a[i]), 1) for i in range(min(len(xp_a), len(xp_b)))]

    fit_a = float(da["fitness"].float().mean().item()) if "fitness" in da else 0.0
    fit_b = float(db["fitness"].float().mean().item()) if "fitness" in db else 0.0

    res = {
        "a": pa.name,
        "b": pb.name,
        "epoch_a": da.get("epoch"),
        "epoch_b": db.get("epoch"),
        "wins_a": da.get("wins_per_faction"),
        "wins_b": db.get("wins_per_faction"),
        "generals_xp_delta": xp_delta,
        "mean_fitness_a": round(fit_a, 3),
        "mean_fitness_b": round(fit_b, 3),
        "mean_fitness_delta": round(fit_b - fit_a, 3),
    }

    if as_json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    print(f"[AEGIS SNAP DIFF] {res['a']} (ep {res['epoch_a']}) -> {res['b']} (ep {res['epoch_b']})")
    print(f"  wins:         {res['wins_a']} -> {res['wins_b']}")
    print(f"  generals_xp:  delta={res['generals_xp_delta']}")
    print(f"  mean_fitness: {res['mean_fitness_a']} -> {res['mean_fitness_b']} ({res['mean_fitness_delta']:+.3f})")
    return 0
