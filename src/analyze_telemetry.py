"""
analyze_telemetry.py - Vědecká Analýza Kampaně, ROI Tříd, Křivky Učení a Prostorových Heatmap
Spuštění:
  python analyze_telemetry.py
  python analyze_telemetry.py --export
  python analyze_telemetry.py --train 15
"""

import os
import json
import glob
import time
import argparse
import numpy as np


def render_ascii_heatmap(events: list[dict], grid_w: int = 44, grid_h: int = 15, max_w: float = 2400.0, max_h: float = 1350.0):
    """Vykreslí textovou prostorovou heatmapu zabití v terminálu."""
    grid = np.zeros((grid_h, grid_w), dtype=np.int32)
    for ev in events:
        gx = int(min(grid_w - 1, max(0, (ev["x"] / max_w) * grid_w)))
        gy = int(min(grid_h - 1, max(0, (ev["y"] / max_h) * grid_h)))
        grid[gy, gx] += 1

    chars = " .:-=+*#%@"
    max_val = max(1, int(np.max(grid)))

    print("\n   [PROSTOROVÁ HEATMAPA BOJIŠTĚ - HUSTOTA ZABITÍ (KILLS)]")
    print("   +" + "-" * grid_w + "+")
    for r in range(grid_h):
        line = ""
        for c in range(grid_w):
            v = grid[r, c]
            idx = int((v / max_val) * (len(chars) - 1))
            line += chars[idx]
        print(f"   |{line}|")
    print("   +" + "-" * grid_w + "+")
    print("   Měřítko intenzity: [nízká .:-=+*#%@ vysoká]")


def run_headless_training(n_epochs: int):
    """Bleskově natrénuje zadaný počet generací na GPU bez grafického okna a uloží snapshot."""
    import torch
    from engine_core import RealisticBattleEngine
    from telemetry_logger import TelemetryManager

    print("=" * 80)
    print(f"   CYBER-TACTICS // RYCHLÝ BEZHLAVÝ TRÉNINK AI NA GPU ({n_epochs} GENERACÍ)")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    engine = RealisticBattleEngine(2400, 1350, device)
    telemetry = TelemetryManager("telemetry")
    engine.telemetry_mgr = telemetry
    engine.configure(400, 4)

    t0 = time.perf_counter()
    target_epoch = engine.current_epoch + n_epochs
    last_printed_epoch = engine.current_epoch

    while engine.current_epoch < target_epoch:
        engine.step(1.0 / 60.0)
        if engine.current_epoch > last_printed_epoch:
            ep_rec = telemetry.epochs_history[-1] if telemetry.epochs_history else {}
            winner = ep_rec.get("winner_team", -1)
            f_names = ["ALPHA", "OMEGA", "SOLAR", "PHANTOM"]
            w_str = f_names[winner] if 0 <= winner < 4 else "REMIZA"
            with engine._engine_lock:
                lvls = "/".join(str(g.level) for g in engine.brains.generals)
            print(
                f"   [✓] Epocha #{last_printed_epoch:02d} dokončena | "
                f"Vítěz: {w_str:<7} | Úrovně generálů: [{lvls}] | "
                f"Důvod: {ep_rec.get('end_reason', '')}"
            )
            last_printed_epoch = engine.current_epoch

    snap_path = telemetry.save_snapshot(engine, custom_tag="headless_trained")
    elapsed = time.perf_counter() - t0
    print("-" * 80)
    print(f"   Hotovo za {elapsed:.2f} s ({elapsed / max(1, n_epochs):.2f} s / epocha).")
    print(f"   Uložen natrénovaný snapshot: {os.path.basename(snap_path)}")
    print("   [TIP] V grafickém simulátoru stiskni 'L' pro okamžité načtení této populace!")
    print("=" * 80)


def analyze_all():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, default="")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--train", type=int, default=0)
    args = parser.parse_args()

    if args.train > 0:
        run_headless_training(args.train)
        return

    telemetry_dir = "telemetry"
    if not os.path.exists(telemetry_dir):
        print(f"[!] Adresář {telemetry_dir} neexistuje.")
        return

    log_files = glob.glob(os.path.join(telemetry_dir, "telemetry_session_*.jsonl"))
    if not log_files:
        print("[!] Žádné logy telemetrie nebyly nalezeny.")
        return

    # Vybrat nejnovější neprázdný log s alespoň 1 epochou (nebo nejnovější celkově)
    log_files.sort(reverse=True)
    target_log = log_files[0]
    if args.session:
        for f in log_files:
            if args.session in f:
                target_log = f
                break
    else:
        for f in log_files:
            if os.path.getsize(f) > 300:
                target_log = f
                break

    print("=" * 80)
    print("   CYBER-TACTICS // ANALÝZA VÝUKOVÉ TELEMETRIE, ROI & EVOLUCE AI")
    print("=" * 80)
    print(f"\n>> Analyzuji session: {os.path.basename(target_log)}")

    records = []
    with open(target_log, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    session_start = next((r for r in records if r.get("type") == "SESSION_START"), None)
    if session_start:
        print(f"   Spuštěno: {session_start.get('timestamp')}")
        print(f"   Hardware: {session_start.get('device')}")

    epochs = [r for r in records if r.get("type") == "EPOCH_COMPLETED"]
    snapshots = [r for r in records if r.get("type") == "SNAPSHOT_SAVED"]

    print(f"\n>> Celkem zaznamenaných bitev v logu: {len(epochs)}")
    print(f">> Celkem uložených snapshotů (.pt):   {len(snapshots)}")

    faction_names = ["ALPHA VANGUARD", "OMEGA SYNDICATE", "SOLAR CORE", "PHANTOM UNIT"]

    if epochs:
        last_ep = epochs[-1]
        first_ep = epochs[0]
        trophies = last_ep.get("total_wins", [0, 0, 0, 0])

        print("\n" + "-" * 80)
        print("   BILANCE TROFEJÍ V KAMPAŇOVÝCH VÍTĚZSTVÍCH:")
        print("-" * 80)
        for f_id in range(len(trophies)):
            f_name = faction_names[f_id] if f_id < len(faction_names) else f"Frakce {f_id}"
            bar = "#" * trophies[f_id]
            print(f"   {f_name:<16}: {trophies[f_id]:3d} výher  {bar}")

        print("\n" + "-" * 80)
        print("   POSLEDNÍCH 5 ODEHRANÝCH BITEV V KAMPAŇOVÉ ROTACI:")
        print("-" * 80)
        print(f"   {'Epocha':<8} {'Bojiště (Scénář)':<30} {'Čas':<8} {'Vítěz':<16} {'Důvod ukončení'}")
        print("   " + "-" * 76)
        for ep in epochs[-5:]:
            e_num = f"#{ep.get('epoch', 0):02d}"
            scen = ep.get("scenario", "N/A").split("(")[0].strip()[:28]
            dur_sec = ep.get("duration_sec", round(ep.get("duration_ticks", 0) / 60.0, 1))
            time_str = f"{dur_sec:.1f} s"
            w_team = ep.get("winner_team", -1)
            w_str = faction_names[w_team] if (0 <= w_team < len(faction_names)) else "REMIZA"
            reason = ep.get("end_reason", "N/A")
            print(f"   {e_num:<8} {scen:<30} {time_str:<8} {w_str:<16} {reason}")

        # Stav generálů (Level, XP a naučené preference tříd)
        print("\n" + "-" * 80)
        print("   STAV VELITELŮ, ZKUŠENOSTI (XP) A NAUČENÉ PREFERENCE NÁKUPU:")
        print("-" * 80)
        f_stats = last_ep.get("factions", [])
        f_stats_first = first_ep.get("factions", [])
        for idx, fs in enumerate(f_stats):
            fid = fs.get("faction_id", idx)
            fname = faction_names[fid] if fid < len(faction_names) else f"Frakce {fid}"
            lvl = fs.get("level", 0)
            xp = int(fs.get("xp", 0))
            fit_avg = fs.get("fitness_avg", 0.0)
            fit_start = f_stats_first[idx].get("fitness_avg", 0.0) if idx < len(f_stats_first) else 0.0
            fit_delta = fit_avg - fit_start
            probs = fs.get("class_probs", [1.0, 0.0, 0.0, 0.0])
            p_str = f"P:{int(probs[0]*100):2d}% M:{int(probs[1]*100):2d}% S:{int(probs[2]*100):2d}% J:{int(probs[3]*100):2d}%"
            print(
                f"   • {fname:<15}: L{lvl} ({xp:5d} XP) | Prům.Fit: {fit_avg:5.1f} (d {fit_delta:+5.1f}) | Váhy: [{p_str}]"
            )

        # ROI Analýza tříd
        roi_list = last_ep.get("class_roi", [])
        if roi_list:
            print("\n" + "-" * 80)
            print("   ROI ANALÝZA TŘÍD JEDNOTEK (NÁVRATNOST NA 1 BOD ROZPOČTU):")
            print("-" * 80)
            print(f"   {'Třída':<12} {'Cena':<6} {'Počet':<7} {'Prům.Fit':<10} {'ROI / 1b':<10} {'Dmg celkem':<12} {'Vyléčeno HP'}")
            print("   " + "-" * 76)
            for r in roi_list:
                print(
                    f"   {r.get('class_name', ''):<12} "
                    f"{r.get('cost', 1)} b   "
                    f"{r.get('count', 0):<7d} "
                    f"{r.get('avg_fitness', 0.0):<10.1f} "
                    f"{r.get('roi_per_point', 0.0):<10.2f} "
                    f"{r.get('total_damage', 0.0):<12.0f} "
                    f"{r.get('total_healing', 0.0):.0f}"
                )

    session_id = target_log.split("telemetry_session_")[-1].replace(".jsonl", "")
    parquet_file = os.path.join(telemetry_dir, f"spatial_events_{session_id}.parquet")
    if not os.path.exists(parquet_file):
        all_parquets = sorted(glob.glob(os.path.join(telemetry_dir, "spatial_events_*.parquet")), reverse=True)
        if all_parquets:
            parquet_file = all_parquets[0]

    if os.path.exists(parquet_file):
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_file)
            print("\n" + "-" * 80)
            print(f"   PROSTOROVÁ TELEMETRIE ({os.path.basename(parquet_file)}): {len(df)} bojových událostí")
            render_ascii_heatmap(df.to_dict(orient="records"))
            if args.export:
                csv_out = os.path.join(telemetry_dir, f"export_spatial_{session_id}.csv")
                df.to_csv(csv_out, index=False, encoding="utf-8")
                print(f"\n   [EXPORT] Prostorová data uložena do: {csv_out}")
        except Exception as e:
            print(f"   [!] Chyba při čtení Parquet souboru: {e}")

    if args.export and epochs:
        try:
            import pandas as pd
            rows = []
            for ep in epochs:
                for fs in ep.get("factions", []):
                    rows.append({
                        "epoch": ep.get("epoch"),
                        "scenario": ep.get("scenario"),
                        "winner_team": ep.get("winner_team"),
                        "faction_id": fs.get("faction_id"),
                        "level": fs.get("level"),
                        "xp": fs.get("xp"),
                        "kills": fs.get("kills"),
                        "node_points": fs.get("node_points"),
                        "fitness_avg": fs.get("fitness_avg"),
                        "fitness_max": fs.get("fitness_max"),
                    })
            df_ep = pd.DataFrame(rows)
            ep_csv = os.path.join(telemetry_dir, f"export_epochs_{session_id}.csv")
            ep_pq = os.path.join(telemetry_dir, f"export_epochs_{session_id}.parquet")
            df_ep.to_csv(ep_csv, index=False, encoding="utf-8")
            df_ep.to_parquet(ep_pq, index=False)
            print(f"   [EXPORT] Historie učení epoch uložena do: {ep_csv} a {ep_pq}")
        except Exception as e:
            print(f"   [!] Chyba při exportu tabulek: {e}")

    snap_files = glob.glob(os.path.join(telemetry_dir, "snapshot_*.pt"))
    if snap_files:
        snap_files.sort(reverse=True)
        print("\n" + "-" * 80)
        print("   DOSTUPNÉ SNAPSHOTY KAMPANĚ PRO OBNOVENÍ:")
        print("-" * 80)
        for sf in snap_files[:5]:
            sz_mb = os.path.getsize(sf) / (1024 ** 2)
            print(f"   • {os.path.basename(sf)} ({sz_mb:.2f} MB)")
        print("\n   [TIP] Stiskem 'L' v simulátoru obnovíte přesný stav generálů, XP i všech 4 bank mozků!")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_all()
