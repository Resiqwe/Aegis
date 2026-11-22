"""
devtools/hub.py - Lehký interaktivní TUI rozcestník vývojářských nástrojů Cyber-Tactics 2.0.

Používá výhradně standardní knihovnu Pythonu. Umožňuje jedním stiskem klávesy spouštět:
  1. HW Probe  – živý monitor zdrojů (devtools/hw_probe.py)
  2. X-Ray     – statická AST analýza kódu (scripts/aegis_tool.py xray --compact)
  3. Profiler  – sekční měření simulačního tiku (scripts/aegis_tool.py profile --compact)
  4. Benchmark – deterministický benchmark a kontrakty (scripts/aegis_tool.py bench)
  5. Health    – kompletní pre-commit kontrola (scripts/aegis_tool.py health)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
HW_PROBE = ROOT_DIR / "devtools" / "hw_probe.py"
AEGIS_TOOL = ROOT_DIR / "scripts" / "aegis_tool.py"


def _configure_stdio() -> None:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def _run_tool(cmd: list[str]) -> int:
    print()
    try:
        return subprocess.run(cmd, cwd=str(ROOT_DIR)).returncode
    except KeyboardInterrupt:
        print("\n[Hub] Běh nástroje přerušen uživatelem (Ctrl+C).")
        return 0


def _prompt(label: str) -> str | None:
    try:
        return input(label).strip()
    except (KeyboardInterrupt, EOFError):
        return None


def _run_hw_probe_menu() -> None:
    tag = _prompt("  Volitelný popisek měření (tag) [Enter = bez tagu]: ")
    if tag is None:
        return
    dur = _prompt("  Limit délky v sekundách [Enter = běží do Ctrl+C]: ")
    if dur is None:
        return

    cmd = [sys.executable, str(HW_PROBE), "--auto"]
    if tag:
        cmd.extend(["--tag", tag])
    if dur:
        cmd.extend(["--duration", dur])
    _run_tool(cmd)


def _print_menu() -> None:
    print("\n" + "=" * 64)
    print("  CYBER-TACTICS 2.0 // DEVTOOLS HUB")
    print("=" * 64)
    print("  1. HW Probe      – živý monitor zdrojů (CPU / RAM / GPU / PCIe)")
    print("  2. X-Ray         – statická AST analýza kódu (--compact)")
    print("  3. Profiler      – sekční měření simulačního tiku (--compact)")
    print("  4. Benchmark     – deterministický benchmark & kontrakty")
    print("  5. Health        – kompletní kontrola (compile + xray + testy)")
    print("  6. Bench Compare – porovnání proti scripts/baseline_bench.json")
    print("  7. Snaps List    – přehled uložených .pt snapshotů")
    print("  0. Konec")
    print("-" * 64)


def main() -> int:
    _configure_stdio()
    try:
        while True:
            _print_menu()
            choice = _prompt("Vyber akci [0-7]: ")
            if choice is None or choice in ("0", "q", "exit"):
                print("Ukončuji DevTools Hub.")
                return 0
            if choice == "1":
                _run_hw_probe_menu()
            elif choice == "2":
                _run_tool([sys.executable, str(AEGIS_TOOL), "xray", "--compact"])
            elif choice == "3":
                _run_tool([sys.executable, str(AEGIS_TOOL), "profile", "--compact"])
            elif choice == "4":
                _run_tool([sys.executable, str(AEGIS_TOOL), "bench"])
            elif choice == "5":
                _run_tool([sys.executable, str(AEGIS_TOOL), "health"])
            elif choice == "6":
                _run_tool([sys.executable, str(AEGIS_TOOL), "bench", "--compare"])
            elif choice == "7":
                _run_tool([sys.executable, str(AEGIS_TOOL), "snaps", "list"])
            else:
                print(f"Neznámá volba '{choice}'. Zadej číslo 0 až 7.")
    except KeyboardInterrupt:
        print("\nUkončuji DevTools Hub.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
