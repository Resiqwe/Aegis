"""
scripts/aegis_tool.py - Jednotný CLI nástroj pro diagnostiku, profiling, benchmark a health-check
projektu Cyber-Tactics 2.0 (Aegis Engine).

Použití:
  python scripts/aegis_tool.py xray [--compact] [--deps ClassName] [--json] [--strict]
  python scripts/aegis_tool.py profile [--compact] [--ticks 200] [--units 800] [--json]
  python scripts/aegis_tool.py bench [--save-baseline] [--compare] [--ticks 500] [--json]
  python scripts/aegis_tool.py snaps list | inspect <file.pt> | diff <a.pt> <b.pt>
  python scripts/aegis_tool.py health
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tools.xray import run_xray, _iter_active_py_files
from tools.perf_runner import run_profile, run_bench
from tools.snap_inspector import list_snapshots, inspect_snapshot, diff_snapshots


def _run_health(root: Path) -> int:
    print("=== [1/4] PY_COMPILE ===")
    py_files = [str(p) for _, p in _iter_active_py_files(root)]
    comp = subprocess.run([sys.executable, "-m", "py_compile", *py_files], cwd=str(root))
    if comp.returncode != 0:
        print("[HEALTH FAIL] py_compile selhal.")
        return 1
    print(f"  OK ({len(py_files)} souborů)")

    print("\n=== [2/4] XRAY --STRICT ===")
    xr_code = run_xray(root, compact=True, strict=True)
    if xr_code != 0:
        print("[HEALTH FAIL] xray --strict nalezl F401 nebo zakázaný GPU sync v core hot-loopu.")
        return 1

    print("\n=== [3/4] TEST SUITE (tests/test_suite.py) ===")
    t_proc = subprocess.run([sys.executable, str(root / "tests" / "test_suite.py"), "-v"], cwd=str(root))
    if t_proc.returncode != 0:
        print("[HEALTH FAIL] test_suite.py selhal.")
        return 1

    print("\n=== [4/4] GIT STATUS ===")
    g_proc = subprocess.run(["git", "status", "--short"], cwd=str(root), capture_output=True, text=True)
    out = (g_proc.stdout or "").strip()
    print(out if out else "  (čistý pracovní strom)")

    print("\n[AEGIS HEALTH] ALL CHECKS PASSED (exit=0)")
    return 0


def main() -> int:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(prog="aegis_tool", description="Aegis Engine Diagnostic & Agent Tooling Kit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # xray
    p_xray = sub.add_parser("xray", help="Statický AST rentgen kódu")
    p_xray.add_argument("--compact", action="store_true", help="Kompaktní výstup pro LLM")
    p_xray.add_argument("--json", action="store_true", help="JSON výstup")
    p_xray.add_argument("--deps", type=str, default=None, help="Mapa READS/WRITES atributů pro zadanou třídu")
    p_xray.add_argument("--strict", action="store_true", help="Exit code 1 při F401 nebo core hot-loop GPU syncu")
    p_xray.add_argument("--strict-all", action="store_true", help="Exit code 1 při F401 nebo jakémkoliv hot-loop syncu")

    # profile
    p_prof = sub.add_parser("profile", help="Sekční GPU/CPU profiler simulačního tiku")
    p_prof.add_argument("--compact", action="store_true", help="Kompaktní výstup pro LLM")
    p_prof.add_argument("--json", action="store_true", help="JSON výstup")
    p_prof.add_argument("--units", type=int, default=800, help="Počet jednotek (default 800)")
    p_prof.add_argument("--warmup", type=int, default=30, help="Počet warm-up tiků (default 30)")
    p_prof.add_argument("--ticks", type=int, default=200, help="Počet měřených tiků (default 200)")
    p_prof.add_argument("--seed", type=int, default=42, help="Seed (default 42)")

    # bench
    p_bench = sub.add_parser("bench", help="Deterministický benchmark + kontrola kontraktů")
    p_bench.add_argument("--save-baseline", action="store_true", help="Uloží výsledek do scripts/baseline_bench.json")
    p_bench.add_argument("--compare", action="store_true", help="Porovná proti scripts/baseline_bench.json")
    p_bench.add_argument("--json", action="store_true", help="JSON výstup")
    p_bench.add_argument("--units", type=int, default=800, help="Počet jednotek (default 800)")
    p_bench.add_argument("--ticks", type=int, default=500, help="Počet tiků (default 500)")
    p_bench.add_argument("--seed", type=int, default=42, help="Seed (default 42)")

    # snaps
    p_snaps = sub.add_parser("snaps", help="Inspektor uložených .pt snapshotů")
    snaps_sub = p_snaps.add_subparsers(dest="snaps_cmd", required=True)
    ps_list = snaps_sub.add_parser("list", help="Seznam snapshotů v telemetry/")
    ps_list.add_argument("--dir", type=str, default="telemetry")
    ps_list.add_argument("--json", action="store_true")

    ps_insp = snaps_sub.add_parser("inspect", help="Detailní inspekce .pt snapshotu")
    ps_insp.add_argument("path", type=str)
    ps_insp.add_argument("--json", action="store_true")

    ps_diff = snaps_sub.add_parser("diff", help="Porovnání dvou .pt snapshotů")
    ps_diff.add_argument("path_a", type=str)
    ps_diff.add_argument("path_b", type=str)
    ps_diff.add_argument("--json", action="store_true")

    # health
    sub.add_parser("health", help="Kompletní pre-commit gate (py_compile + xray + tests + git)")

    args = parser.parse_args()

    if args.cmd == "xray":
        return run_xray(
            ROOT,
            compact=args.compact,
            as_json=args.json,
            deps_class=args.deps,
            strict=args.strict,
            strict_all=args.strict_all,
        )
    if args.cmd == "profile":
        return run_profile(
            ROOT,
            n_units=args.units,
            warmup=args.warmup,
            ticks=args.ticks,
            seed=args.seed,
            compact=args.compact,
            as_json=args.json,
        )
    if args.cmd == "bench":
        return run_bench(
            ROOT,
            n_units=args.units,
            ticks=args.ticks,
            seed=args.seed,
            save_baseline=args.save_baseline,
            compare=args.compare,
            as_json=args.json,
        )
    if args.cmd == "snaps":
        if args.snaps_cmd == "list":
            return list_snapshots(ROOT, telemetry_dir=args.dir, as_json=args.json)
        if args.snaps_cmd == "inspect":
            return inspect_snapshot(args.path, as_json=args.json)
        if args.snaps_cmd == "diff":
            return diff_snapshots(args.path_a, args.path_b, as_json=args.json)
    if args.cmd == "health":
        return _run_health(ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
