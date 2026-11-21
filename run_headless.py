"""
run_headless.py - Tichý Běh Simulace AI na Pozadí (100% Headless / Bez Vykreslování Okna)
Umožňuje trénovat armády a neuro-evoluci na pozadí bez nutnosti otevírat jakékoliv grafické okno:
  1. Běží na GPU v čistém výpočetním režimu s minimální spotřebou systémových zdrojů (kooperativní yielding).
  2. Automaticky zapisuje průběžnou telemetrii (JSONL, Parquet) a každých 10 epoch ukládá .pt snapshot.
  3. Lze spustit jedním klikem přes Spustit_Na_Pozadi_Headless.vbs nebo z příkazové řádky.
  4. Grafický Launcher (launcher.py) živě čte a zobrazuje data produkovaná tímto procesem.
"""

import os
import sys
import time
import signal
import argparse
import glob

# Retenční limit: kolik nejnovějších snapshotů a parquet souborů nechat
MAX_SNAPSHOTS_TO_KEEP = 10
MAX_PARQUET_FILES_TO_KEEP = 5

# Ochrana pro běh pod pythonw.exe
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import engine_core
import telemetry_logger

RUNNING = True


def signal_handler(sig, frame):
    global RUNNING
    print("\n[!] Zachycen signál k ukončení. Dokončuji aktuální krok a ukládáme snapshot...")
    RUNNING = False


def cleanup_old_telemetry(telemetry_dir: str, max_snapshots: int = MAX_SNAPSHOTS_TO_KEEP,
                          max_parquet: int = MAX_PARQUET_FILES_TO_KEEP):
    """Odstraní staré soubory telemetrie, ponechá jen nejnovější maximální počet."""
    if not os.path.exists(telemetry_dir):
        return

    # Smazání starých snapshotů .pt
    snaps = sorted(
        [f for f in os.listdir(telemetry_dir) if f.startswith("snapshot_") and f.endswith(".pt")],
        key=lambda p: os.path.getmtime(os.path.join(telemetry_dir, p))
    )
    while len(snaps) > max_snapshots:
        old_snap = snaps.pop(0)
        old_path = os.path.join(telemetry_dir, old_snap)
        try:
            os.remove(old_path)
        except Exception:
            pass

    # Smazání starých parquet souborů
    pq_files = sorted(
        [f for f in os.listdir(telemetry_dir) if f.startswith("spatial_events_") and f.endswith(".parquet")],
        key=lambda p: os.path.getmtime(os.path.join(telemetry_dir, p))
    )
    while len(pq_files) > max_parquet:
        old_pq = pq_files.pop(0)
        old_path = os.path.join(telemetry_dir, old_pq)
        try:
            os.remove(old_path)
        except Exception:
            pass


def main():
    global RUNNING
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Cyber-Tactics Headless Background Simulator")
    parser.add_argument("--epochs", type=int, default=-1, help="Počet generací k odehrání (-1 = kontinuální non-stop běh)")
    parser.add_argument("--units", type=int, default=400, help="Celkový rozpočet jednotek (200, 400, 600, 800)")
    parser.add_argument("--scenario", type=int, default=0, help="Výchozí scénář (0=Domination, 1=Citadel, 2=Chasm)")
    parser.add_argument("--load", type=str, default="", help="Cesta ke snapshotu .pt pro navázání")
    parser.add_argument("--telemetry-dir", type=str, default="telemetry", help="Složka pro telemetrii")
    args, _ = parser.parse_known_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(args.telemetry_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("   ⚡ CYBER-TACTICS // HEADLESS BACKGROUND SIMULATOR")
    print("=" * 78)
    print(f"   Zařízení:       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"   Režim:          {'Kontinuální (Non-stop)' if args.epochs == -1 else f'{args.epochs} generací'}")
    print(f"   Rozpočet:       {args.units} jednotek")
    print(f"   Telemetrie:     {os.path.abspath(args.telemetry_dir)}")

    engine = engine_core.RealisticBattleEngine(2400, 1350, device)
    # Headless běh nepotřebuje tracers, lasery ani částicové efekty.
    engine.visual_mode = False
    telemetry = telemetry_logger.TelemetryManager(args.telemetry_dir)
    engine.telemetry_mgr = telemetry
    engine.configure(args.units, 4)

    if args.scenario in (0, 1, 2):
        engine.switch_scenario(args.scenario)

    # Načtení snapshotu
    target_snap = args.load
    if not target_snap:
        snaps = glob.glob(os.path.join(args.telemetry_dir, "snapshot_*.pt"))
        snaps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        if snaps:
            target_snap = snaps[0]

    if target_snap and os.path.exists(target_snap):
        if telemetry.load_snapshot(engine, target_snap):
            print(f"   Načten model:   {os.path.basename(target_snap)}")

    print("-" * 78)
    print("   [+] Simulace spuštěna na pozadí. Pro ukončení stiskněte Ctrl+C.")
    print("-" * 78)

    start_epoch = engine.current_epoch
    last_epoch = engine.current_epoch
    done_epochs = 0
    t0 = time.perf_counter()

    step_dt = 1.0 / 60.0
    tick_in_sec = 0

    try:
        while RUNNING:
            engine.step(step_dt)
            tick_in_sec += 1

            # Každých 60 kroků krátký sleep pro kooperativní ochranu CPU a chatu
            if tick_in_sec >= 60:
                time.sleep(0.001)
                tick_in_sec = 0

            if engine.current_epoch > last_epoch:
                done_epochs += 1
                rec = telemetry.epochs_history[-1] if telemetry.epochs_history else {}
                w_team = rec.get("winner_team", -1)
                w_name = ["ALPHA", "OMEGA", "SOLAR", "PHANTOM"][w_team] if 0 <= w_team < 4 else "REMIZA"
                dur = rec.get("duration_ticks", 0) / 60.0
                reason = rec.get("end_reason", "Vítězství")
                with engine._engine_lock:
                    lvls = "/".join(str(g.level) for g in engine.brains.generals)

                print(
                    f"\r   [Ep #{last_epoch:03d} -> #{engine.current_epoch:03d}] "
                    f"Vítěz: {w_name:<7} | Čas: {dur:4.1f}s | Levely: [{lvls}] | {reason}",
                    flush=True,
                )
                last_epoch = engine.current_epoch
                telemetry.flush_spatial_parquet()

                # Periodický auto-save každých 10 epoch
                if done_epochs % 10 == 0:
                    snap_path = telemetry.save_snapshot(engine, custom_tag="headless_daemon")
                    print(f"   >>> [AUTO-SAVE #{done_epochs:03d}] Uložen model: {os.path.basename(snap_path)}")

                    # Periodická úklid starých souborů (udržujeme pouze nejnovějších N kusů)
                    if done_epochs > 0 and done_epochs % 50 == 0:
                        cleanup_old_telemetry(args.telemetry_dir)

                if args.epochs > 0 and done_epochs >= args.epochs:
                    print(f"\n[✓] Dosaženo cílového počtu {args.epochs} generací.")
                    break

    except KeyboardInterrupt:
        print("\n[!] Přerušeno uživatelem.")

    finally:
        telemetry.flush_spatial_parquet()
        final_snap = telemetry.save_snapshot(engine, custom_tag="headless_final")
        elapsed = time.perf_counter() - t0
        print("-" * 78)
        print(f"   Běh ukončen: Odehráno {done_epochs} generací za {elapsed:.1f}s.")
        print(f"   Konečný snapshot: {os.path.basename(final_snap)}")
        print("=" * 78)


if __name__ == "__main__":
    main()
