"""
hardware_telemetry.py - Komplexní telemetrie vytížení PC vs. systémových nároků simulátoru
Sleduje v reálném čase a ukládá data pro hloubkovou optimalizaci:
  1. PC vs. Program CPU vytížení (%)
  2. PC vs. Program RAM paměť (GB / MB)
  3. GPU Core Utilization (%), Teplota (°C), Celková VRAM vs. PyTorch tenzory
  4. Profilace latence: Engine Step (ms), Pygame Render (ms), FPS a TPS
  5. Automatický záznam do Parquet / CSV pro analýzu výkonnostních úzkých hrdel (bottlenecks)
"""

import os
import time
from datetime import datetime
import torch

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    _has_nvml = True
except Exception:
    _has_nvml = False
    _nvml_handle = None


class HardwareTelemetryTracker:
    def __init__(self, output_dir: str = "telemetry", session_id: str | None = None):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.parquet_path = os.path.join(self.output_dir, f"hardware_profiling_{self.session_id}.parquet")
        self.csv_path = os.path.join(self.output_dir, f"hardware_profiling_{self.session_id}.csv")

        self.proc = psutil.Process(os.getpid()) if psutil else None
        if self.proc:
            self.proc.cpu_percent()
        if psutil:
            psutil.cpu_percent()

        self.history = []
        self.recent_samples = []  # Kruhový buffer posledních 60 vzorků pro UI graf
        self.last_sample_time = 0.0
        self.last_flush_time = time.time()

        # Poslední změřený stav
        self.current_metrics = {
            "sys_cpu_pct": 0.0,
            "proc_cpu_pct": 0.0,
            "sys_ram_used_gb": 0.0,
            "sys_ram_total_gb": 0.0,
            "sys_ram_pct": 0.0,
            "proc_ram_mb": 0.0,
            "gpu_util_pct": 0,
            "gpu_temp_c": 0,
            "gpu_mem_used_mb": 0.0,
            "gpu_mem_total_mb": 0.0,
            "torch_vram_alloc_mb": 0.0,
            "torch_vram_res_mb": 0.0,
            "step_ms": 0.0,
            "render_ms": 0.0,
            "ui_fps": 0.0,
            "sim_tps": 0.0,
        }

    def sample(self, step_ms: float = 0.0, render_ms: float = 0.0, ui_fps: float = 0.0, sim_tps: float = 0.0) -> dict:
        """Zaznamená aktuální snímek systémových prostředků (throttlováno na min 250ms)."""
        now = time.time()
        # Vzorkování provádíme každých ~250ms, abychom nezatěžovali CPU měřením
        if (now - self.last_sample_time) < 0.25 and self.history:
            # Pouze aktualizujeme okamžité latence
            self.current_metrics["step_ms"] = round(step_ms, 2)
            self.current_metrics["render_ms"] = round(render_ms, 2)
            self.current_metrics["ui_fps"] = round(ui_fps, 1)
            self.current_metrics["sim_tps"] = round(sim_tps, 1)
            return self.current_metrics

        self.last_sample_time = now

        # 1. CPU
        proc_cpu = 0.0
        sys_cpu = 0.0
        proc_ram_mb = 0.0
        sys_ram_used_gb = 0.0
        sys_ram_total_gb = 0.0
        sys_ram_pct = 0.0

        if psutil and self.proc:
            try:
                proc_cpu = self.proc.cpu_percent() / max(1, psutil.cpu_count())
                proc_ram_mb = self.proc.memory_info().rss / (1024 ** 2)
                sys_cpu = psutil.cpu_percent()
                sys_ram = psutil.virtual_memory()
                sys_ram_used_gb = sys_ram.used / (1024 ** 3)
                sys_ram_total_gb = sys_ram.total / (1024 ** 3)
                sys_ram_pct = sys_ram.percent
            except Exception:
                pass

        # 2. GPU & VRAM
        gpu_util = 0
        gpu_temp = 0
        gpu_mem_used_mb = 0.0
        gpu_mem_total_mb = 0.0

        if _has_nvml and _nvml_handle:
            try:
                rates = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
                gpu_util = rates.gpu
                gpu_temp = pynvml.nvmlDeviceGetTemperature(_nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                mem = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
                gpu_mem_used_mb = mem.used / (1024 ** 2)
                gpu_mem_total_mb = mem.total / (1024 ** 2)
            except Exception:
                pass

        torch_alloc = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        torch_res = torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "time_rel_sec": round(now - (self.history[0]["_raw_time"] if self.history else now), 2),
            "_raw_time": now,
            "sys_cpu_pct": round(sys_cpu, 1),
            "proc_cpu_pct": round(proc_cpu, 1),
            "sys_ram_used_gb": round(sys_ram_used_gb, 2),
            "sys_ram_total_gb": round(sys_ram_total_gb, 2),
            "sys_ram_pct": round(sys_ram_pct, 1),
            "proc_ram_mb": round(proc_ram_mb, 1),
            "gpu_util_pct": int(gpu_util),
            "gpu_temp_c": int(gpu_temp),
            "gpu_mem_used_mb": round(gpu_mem_used_mb, 1),
            "gpu_mem_total_mb": round(gpu_mem_total_mb, 1),
            "torch_vram_alloc_mb": round(torch_alloc, 1),
            "torch_vram_res_mb": round(torch_res, 1),
            "step_ms": round(step_ms, 2),
            "render_ms": round(render_ms, 2),
            "ui_fps": round(ui_fps, 1),
            "sim_tps": round(sim_tps, 1),
        }

        self.current_metrics = metrics
        self.history.append(metrics)
        self.recent_samples.append(metrics)
        if len(self.recent_samples) > 60:
            self.recent_samples.pop(0)

        # Periodický flush každých 30 sekund
        if now - self.last_flush_time > 30.0 and len(self.history) > 10:
            self.flush()

        return metrics

    def flush(self):
        """Uloží nasbíranou telemetrii do Parquet a CSV na disk."""
        if not self.history:
            return
        self.last_flush_time = time.time()
        try:
            import pandas as pd
            # Očistíme privátní pole
            clean_records = [{k: v for k, v in r.items() if not k.startswith("_")} for r in self.history]
            df = pd.DataFrame(clean_records)
            df.to_parquet(self.parquet_path, index=False)
            df.to_csv(self.csv_path, index=False)
        except Exception:
            pass

    def get_optimization_summary(self) -> dict:
        """Vygeneruje expertní analýzu úzkých hrdel programu vs. PC."""
        if not self.history:
            return {"status": "Nedostatek dat pro analýzu"}

        recent = self.history[-40:] if len(self.history) >= 40 else self.history
        avg_step = sum(r["step_ms"] for r in recent) / len(recent)
        avg_render = sum(r["render_ms"] for r in recent) / len(recent)
        avg_gpu = sum(r["gpu_util_pct"] for r in recent) / len(recent)
        avg_proc_cpu = sum(r["proc_cpu_pct"] for r in recent) / len(recent)
        avg_sys_cpu = sum(r["sys_cpu_pct"] for r in recent) / len(recent)
        max_vram_alloc = max(r["torch_vram_alloc_mb"] for r in recent)
        cur_proc_ram = recent[-1]["proc_ram_mb"]

        bottlenecks = []
        recommendations = []

        if avg_step > 16.6:
            bottlenecks.append(f"Krok simulace ({avg_step:.1f}ms) překračuje 60 FPS frame-budget (16.6ms)")
            recommendations.append("Doporučeno: Využít GPU spatial hash a zredukovat redundantní cdist matice.")
        else:
            recommendations.append(f"Fyzika a neuronové sítě běží bleskově ({avg_step:.1f}ms/krok, rezerva do 60 FPS).")

        if avg_gpu < 35:
            recommendations.append(f"GPU jádro má masivní rezervu ({100 - avg_gpu:.0f}% volno). Lze zdesetinásobit armády nebo zapnout paralelní multi-battle trénink.")

        if avg_render > 12.0:
            bottlenecks.append(f"Vykreslování rozhraní trvá {avg_render:.1f}ms – blíží se limitu plynulosti.")
            recommendations.append("Doporučeno: Omezit překreslování radarových prvků na 30 FPS nebo využít Dirty Rects.")

        return {
            "samples_count": len(self.history),
            "avg_step_ms": round(avg_step, 2),
            "avg_render_ms": round(avg_render, 2),
            "avg_gpu_util_pct": round(avg_gpu, 1),
            "avg_proc_cpu_pct": round(avg_proc_cpu, 1),
            "avg_sys_cpu_pct": round(avg_sys_cpu, 1),
            "max_tensor_vram_mb": round(max_vram_alloc, 1),
            "current_proc_ram_mb": round(cur_proc_ram, 1),
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
        }
