#!/usr/bin/env python3
"""Cyber-Tactics Project Structure Reorganization"""
import os
import shutil
import pathlib

PROJECT = pathlib.Path.cwd()

# ── Source → Destination mapping (idempotent) ───────────────────
MOVES = [
    # Unify engine/ and engine_core.py into src/engine package
    (PROJECT / "engine_core.py", PROJECT / "src" / "engine_core.py"),
    # Package submodules remain at engine/ subdir
    (PROJECT / "engine" / "__init__.py", PROJECT / "src" / "engine" / "__init__.py"),
    # Launcher and AI brains into src/
    (PROJECT / "launcher.py", PROJECT / "src" / "launcher.py"),
    (PROJECT / "ai_brains.py", PROJECT / "src" / "ai_brains.py"),
    # Scratch → archive (preserve history)
    (PROJECT / "scratch" / "final_telemetry_check.py", PROJECT / "archive" / "scratch"),
    (PROJECT / "scratch" / "test_aegis_async.py", PROJECT / "archive" / "scratch"),
    (PROJECT / "scratch" / "verify_speed_telemetry.py", PROJECT / "archive" / "scratch"),
]

def organize():
    for src, dst in MOVES:
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Idempotent: only move if source exists AND dest doesn't
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
        elif dst.exists():
            pass  # Already there, skip
        else:
            print(f"SKIP (src missing): {src.relative_to(PJECT)}")

    # Clean __pycache__ recursively
    for pyc in PROJECT.rglob("__pycache__"):
        shutil.rmtree(str(pyc))

    # Consolidate archiv subdirs safely
    archiv = PROJECT / "archiv"
    if archiv.exists():
        for sub in ["_historie", "arena_a_prototypy", "legacy_benchmarks"]:
            p = archiv / sub
            if p.exists() and p.is_dir():
                (PROJECT / "archive" / sub).mkdir(exist_ok=True)
                for item in p.rglob("*"):
                    shutil.move(str(item), str(PROJECT / "archive" / sub / item.name))
                shutil.rmtree(str(p))


if __name__ == "__main__":
    organize()
    print("Reorganization complete.")