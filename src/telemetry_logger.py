"""
telemetry_logger.py - Vědecká Telemetrie s Prostorovými Heatmapami, ROI Analýzou a Správou Snapshotů
Ukládá a analyzuje:
  1. Hloubková prostorová telemetrie (Kill & Damage Heatmapy):
     - Přesné souřadnice (x, y) každého zabití, frakce a třídy útočníka i oběti.
  2. ROI Analýza armádního rozpočtu (100 bodů):
     - Efektivita každé třídy (udělené zranění, vyléčené HP, přežití a průměrná fitness na 1 bod ceny).
  3. Bleskový export do Parquet / JSONL / CSV pro analýzu a plynulé ukládání/načítání celých genetických bank (.pt).
"""

import os
import json
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _json_serial_default(o):
    """Zaručí bezpečný převod NumPy a PyTorch typů (např. float32, int64, ndarray) do nativních JSON typů."""
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _copy_compatible_weights(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """Zkopíruje kompatibilní podmnožinu vah (adaptér pro staré snapshoty 10x24 -> 14x28).

    Pokud se tvary liší, zkopíruje se průnik všech dimenzí. Zbývající hodnoty
    v cíli zůstanou nedotčené (čerstvá inicializace), takže načtení nespadne
    na RuntimeError a starý model se dá dále dotrénovat.
    """
    if dst.shape == src.shape and dst.dtype == src.dtype and dst.device == src.device:
        dst.copy_(src)
        return True
    src = src.to(device=dst.device, dtype=dst.dtype)
    if dst.shape == src.shape:
        dst.copy_(src)
        return True
    slices = tuple(slice(0, min(d, s)) for d, s in zip(dst.shape, src.shape))
    dst[slices] = src[slices]
    return True


class TelemetryManager:
    def __init__(self, output_dir: str = "telemetry"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filepath = os.path.join(self.output_dir, f"telemetry_session_{self.session_id}.jsonl")
        self.summary_filepath = os.path.join(self.output_dir, "latest_summary.json")
        self.spatial_parquet_path = os.path.join(self.output_dir, f"spatial_events_{self.session_id}.parquet")

        self.epochs_history = []
        self.spatial_events = []
        self._header_written = False

    def _ensure_session_header(self):
        if self._header_written:
            return
        self._header_written = True
        header = {
            "type": "SESSION_START",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "notes": "Cyber-Tactics AI Training & Neuro-Evolution Simulator",
        }
        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")

    def record_kill_event(
        self,
        epoch: int,
        time_sec: float,
        killer_team: int,
        victim_team: int,
        killer_class: int,
        victim_class: int,
        x: float,
        y: float,
    ):
        """Zaznamená přesnou prostorovou souřadnici zabití pro heatmapu."""
        self.spatial_events.append({
            "epoch": epoch,
            "time_sec": round(time_sec, 2),
            "killer_team": killer_team,
            "victim_team": victim_team,
            "killer_class": killer_class,
            "victim_class": victim_class,
            "x": round(float(x), 1),
            "y": round(float(y), 1),
        })

    def flush_spatial_parquet(self) -> str | None:
        """Zapíše dosud nasbírané prostorové události do Parquet souboru."""
        if not self.spatial_events:
            return None
        try:
            df = pd.DataFrame(self.spatial_events)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, self.spatial_parquet_path, compression="snappy")
            self.spatial_events.clear()  # Clear after successful write to prevent unbounded growth
            return self.spatial_parquet_path
        except Exception:
            return None

    def get_live_history(self) -> list[dict]:
        """Vrátí historii epoch přímo z RAM (bez čtení JSONL z disku)."""
        return self.epochs_history

    def build_heatmap_grid(self, grid_w: int = 36, grid_h: int = 22,
                           map_w: float = 1600.0, map_h: float = 1000.0) -> np.ndarray:
        """Sestaví 2D heatmapu zabití z prostorových událostí držených v RAM."""
        grid = np.zeros((grid_h, grid_w), dtype=np.int32)
        for ev in self.spatial_events:
            xi = int(min(grid_w - 1, max(0, (ev.get("x", 0.0) / map_w) * grid_w)))
            yi = int(min(grid_h - 1, max(0, (ev.get("y", 0.0) / map_h) * grid_h)))
            grid[yi, xi] += 1
        return grid

    def log_epoch(
        self,
        epoch_num: int,
        scenario_name: str,
        duration_ticks: int,
        winner_team: int,
        end_reason: str,
        factions_stats: list[dict],
        total_wins: list[int],
        class_roi: list[dict] | None = None,
    ):
        """Zapíše detailní telemetrii ukončené epochy včetně ROI a odemčení."""
        self._ensure_session_header()
        duration_sec = round(duration_ticks / 60.0, 2)
        record = {
            "type": "EPOCH_COMPLETED",
            "epoch": epoch_num,
            "timestamp": datetime.now().isoformat(),
            "scenario": scenario_name,
            "duration_ticks": duration_ticks,
            "duration_sec": duration_sec,
            "winner_team": winner_team,
            "end_reason": end_reason,
            "total_wins": total_wins,
            "factions": factions_stats,
            "class_roi": class_roi or [],
        }
        self.epochs_history.append(record)
        # Omezte historii epoch na posledních 100 generací (prevence paměťového nárůstu)
        if len(self.epochs_history) > 100:
            self.epochs_history = self.epochs_history[-100:]

        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_serial_default) + "\n")

        summary = {
            "last_updated": datetime.now().isoformat(),
            "session_id": self.session_id,
            "total_epochs": epoch_num,
            "factions_trophies": total_wins,
            "recent_epochs": self.epochs_history[-10:],
        }
        with open(self.summary_filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_serial_default)

        if len(self.spatial_events) > 0 and (epoch_num % 2 == 0 or len(self.spatial_events) > 250):
            self.flush_spatial_parquet()

    def save_snapshot(
        self,
        engine,
        custom_tag: str = "",
    ) -> str:
        """Uloží kompletní stav kampaně, všech 4 genetických bank mozků a generálů do binárního souboru .pt."""
        self._ensure_session_header()
        tag_part = f"_{custom_tag}" if custom_tag else ""
        filename = f"snapshot_{self.session_id}_epoch_{engine.current_epoch:03d}{tag_part}.pt"
        filepath = os.path.join(self.output_dir, filename)

        snapshot_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "epoch": engine.current_epoch,
            "scenario_idx": engine.current_scenario_idx,
            "n_units": engine.N,
            "n_factions": engine.n_factions,
            "budget_per_faction": getattr(engine, "budget_per_faction", 100),
            "wins_per_faction": engine.epoch_wins_per_faction,
            "faction_doctrines": engine.faction_doctrines,
            # Stav generálů (XP, Level, kompoziční preference, EMA ROI)
            "generals_xp": [float(g.xp) for g in engine.brains.generals],
            "generals_level": [int(g.level) for g in engine.brains.generals],
            "generals_weights": [g.composition_weights.cpu() for g in engine.brains.generals],
            "generals_roi_ema": [list(g.class_roi_ema) for g in engine.brains.generals],
            # Kompletní genetické banky všech 4 frakcí (MAX_UNITS = 800)
            "brains_W1_full": engine.brains.W1.cpu(),
            "brains_b1_full": engine.brains.b1.cpu(),
            "brains_W2_full": engine.brains.W2.cpu(),
            "brains_b2_full": engine.brains.b2.cpu(),
            "brains_W3_full": getattr(engine.brains, 'W3', engine.brains.W2).cpu(),
            "brains_b3_full": getattr(engine.brains, 'b3', engine.brains.b2).cpu(),
            # Zpětná kompatibilita
            "brains_W1": engine.brains.W1[:engine.N].cpu(),
            "brains_b1": engine.brains.b1[:engine.N].cpu(),
            "brains_W2": engine.brains.W2[:engine.N].cpu(),
            "brains_b2": engine.brains.b2[:engine.N].cpu(),
            # Stav populace
            "teams": engine.teams.cpu(),
            "classes": engine.classes.cpu(),
            "fitness": engine.fitness.cpu(),
            "history_best": engine.brains.history_best_fitness,
            "history_avg": engine.brains.history_avg_fitness,
        }

        torch.save(snapshot_data, filepath)
        self.flush_spatial_parquet()

        with open(self.log_filepath, "a", encoding="utf-8") as f:
            log_entry = {
                "type": "SNAPSHOT_SAVED",
                "epoch": engine.current_epoch,
                "filepath": filepath,
                "filename": filename,
                "timestamp": datetime.now().isoformat(),
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return filepath

    def load_snapshot(self, engine, filepath: str) -> bool:
        """Načte uložený snapshot a obnoví stav mozků, generálů i simulace (včetně zpětné kompatibility)."""
        if not os.path.exists(filepath):
            return False

        data = torch.load(filepath, map_location=engine.device, weights_only=False)

        engine.n_factions = data.get("n_factions", 4)
        engine.budget_per_faction = data.get("budget_per_faction", 100)
        engine.current_epoch = data.get("epoch", 1)
        engine.epoch_wins_per_faction = data.get("wins_per_faction", [0, 0, 0, 0])
        engine.faction_doctrines = data.get("faction_doctrines", [0, 0, 0, 0])
        for f, d in enumerate(engine.faction_doctrines):
            engine.faction_doctrine_tensor[f] = d

        # Obnova generálů kampaně
        if "generals_xp" in data:
            for f in range(min(len(engine.brains.generals), len(data["generals_xp"]))):
                engine.brains.generals[f].xp = float(data["generals_xp"][f])
                engine.brains.generals[f].level = int(data["generals_level"][f])
                if "generals_weights" in data:
                    engine.brains.generals[f].composition_weights.copy_(
                        data["generals_weights"][f].to(engine.device)
                    )
                if "generals_roi_ema" in data and f < len(data["generals_roi_ema"]):
                    engine.brains.generals[f].class_roi_ema = list(data["generals_roi_ema"][f])

        # Obnova kompletních genetických bank (nebo migrace staršího snapshotu po týmech)
        if "brains_W1_full" in data:
            M = min(engine.brains.MAX_UNITS, data["brains_W1_full"].shape[0])
            _copy_compatible_weights(engine.brains.W1[:M], data["brains_W1_full"][:M])
            _copy_compatible_weights(engine.brains.b1[:M], data["brains_b1_full"][:M])
            _copy_compatible_weights(engine.brains.W2[:M], data["brains_W2_full"][:M])
            _copy_compatible_weights(engine.brains.b2[:M], data["brains_b2_full"][:M])
            if "brains_W3_full" in data and hasattr(engine.brains, 'W3'):
                _copy_compatible_weights(engine.brains.W3[:M], data["brains_W3_full"][:M])
                _copy_compatible_weights(engine.brains.b3[:M], data["brains_b3_full"][:M])
        elif "brains_W1" in data and "teams" in data:
            old_teams = data["teams"].to(engine.device)
            old_W1 = data["brains_W1"].to(engine.brains.dtype).to(engine.device)
            old_b1 = data["brains_b1"].to(engine.brains.dtype).to(engine.device)
            old_W2 = data["brains_W2"].to(engine.brains.dtype).to(engine.device)
            old_b2 = data["brains_b2"].to(engine.brains.dtype).to(engine.device)
            for f in range(engine.n_factions):
                f_idx = torch.where(old_teams == f)[0]
                cnt = min(f_idx.numel(), engine.brains.SLOTS_PER_FACTION)
                if cnt > 0:
                    s_start = f * engine.brains.SLOTS_PER_FACTION
                    _copy_compatible_weights(engine.brains.W1[s_start:s_start + cnt], old_W1[f_idx[:cnt]])
                    _copy_compatible_weights(engine.brains.b1[s_start:s_start + cnt], old_b1[f_idx[:cnt]])
                    _copy_compatible_weights(engine.brains.W2[s_start:s_start + cnt], old_W2[f_idx[:cnt]])
                    _copy_compatible_weights(engine.brains.b2[s_start:s_start + cnt], old_b2[f_idx[:cnt]])

        engine.brains.history_best_fitness = data.get("history_best", [[] for _ in range(4)])
        engine.brains.history_avg_fitness = data.get("history_avg", [[] for _ in range(4)])
        engine.brains.epoch_number = engine.current_epoch

        scen_idx = data.get("scenario_idx", 0)
        if scen_idx != engine.current_scenario_idx:
            engine.switch_scenario(scen_idx)
        else:
            engine.start_new_operation(restart_all=False)

        return True

    def list_available_snapshots(self) -> list[str]:
        if not os.path.exists(self.output_dir):
            return []
        files = [f for f in os.listdir(self.output_dir) if f.startswith("snapshot_") and f.endswith(".pt")]
        files.sort(reverse=True)
        return [os.path.join(self.output_dir, f) for f in files]
