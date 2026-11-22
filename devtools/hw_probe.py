"""
devtools/hw_probe.py - Samostatná neinvazivní per-process hardwarová sonda pro Cyber-Tactics 2.0.

Sleduje skutečnou spotřebu zdrojů konkrétního procesu (včetně potomků) bez zásahu do engine kódu
a bez zatížení GILu sledovaného procesu:
  - Výchozí běh bez parametrů: automaticky --auto (najde launcher.py / tactical_hud.py / run_headless.py
    a odfiltruje prázdné rodičovské Windows .venv stub procesy)
  - CPU % stromu procesů + rozpad na top vlákna (threads()) + počet vláken + afinita jader
  - Fyzická RAM (RSS) a privátní paměť (USS / Private Bytes)
  - GPU per-PID VRAM (Compute + Graphics NVML), SM %, Memory Controller %, PCIe TX/RX (KB/s & MB/s),
    spotřeba (W) a takty jádra/pamětí (MHz)
  - Duální logování do devtools/logs/:
      * hw_probe_<timestamp>[_<tag>].csv        (strojově čitelná data)
      * hw_probe_<timestamp>[_<tag>]_report.md  (přehledný Markdown report + automatická diagnóza)

Použití:
  python devtools/hw_probe.py
  python devtools/hw_probe.py --duration 10 --tag boj_800
  python devtools/hw_probe.py --pid 12345 --interval 250 --no-log
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

try:
    import pynvml
except ImportError:
    pynvml = None  # type: ignore[assignment]


PROJECT_SCRIPTS = ("launcher.py", "tactical_hud.py", "run_headless.py")
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT_DIR / "devtools" / "logs"
BATCH_FLUSH_SIZE = 10

CSV_FIELDS = [
    "timestamp",
    "elapsed_s",
    "root_pids",
    "proc_count",
    "cpu_total_pct",
    "cpu_norm_pct",
    "num_threads",
    "affinity",
    "top_threads",
    "ram_rss_mb",
    "ram_uss_mb",
    "ram_private_mb",
    "gpu_pid_vram_mb",
    "gpu_sm_util_pct",
    "gpu_mem_ctrl_pct",
    "pcie_tx_kbps",
    "pcie_rx_kbps",
    "gpu_power_w",
    "gpu_core_mhz",
    "gpu_mem_mhz",
]


def _configure_stdio() -> None:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def _sanitize_tag(raw_tag: str | None) -> str:
    if not raw_tag:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_tag.strip()).strip("_")
    return cleaned[:48]


def _format_affinity(cores: list[int]) -> str:
    if not cores:
        return "n/a"
    s = sorted(set(cores))
    if len(s) == s[-1] - s[0] + 1:
        return f"{s[0]}-{s[-1]} ({len(s)}c)"
    return f"{len(s)}c"


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _compute_stats(values: list[float]) -> tuple[float, float, float, float]:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return 0.0, 0.0, 0.0, 0.0
    s = sorted(valid)
    return s[0], sum(s) / len(s), s[-1], _percentile(s, 95.0)


class NVMLProbe:
    """Neinvazivní wrapper nad NVML pro per-PID VRAM a globální GPU metriky."""

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.available = False
        self.gpu_name = "N/A"
        self._handle: Any = None
        self._warned: set[str] = set()

        if pynvml is None:
            self._warn_once("no_pynvml", "pynvml/nvidia-ml-py není nainstalováno – GPU metriky budou nedostupné.")
            return

        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            raw_name = pynvml.nvmlDeviceGetName(self._handle)
            self.gpu_name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
            self.available = True
        except Exception as exc:
            self._warn_once("nvml_init", f"NVML inicializace selhala: {exc}")
            self.available = False

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            print(f"[hw_probe WARN] {msg}", file=sys.stderr)

    def _query_pid_vram_mb(self, target_pids: set[int]) -> float:
        if not self.available or not target_pids:
            return 0.0

        per_pid_bytes: dict[int, int] = {}
        queries = (
            ("compute_procs", "nvmlDeviceGetComputeRunningProcesses"),
            ("graphics_procs", "nvmlDeviceGetGraphicsRunningProcesses"),
        )
        for key, fn_name in queries:
            fn = getattr(pynvml, fn_name, None)
            if fn is None:
                continue
            try:
                procs = fn(self._handle)
            except Exception as exc:
                self._warn_once(key, f"NVML {fn_name} nedostupné: {exc}")
                continue
            for p in procs:
                pid = int(getattr(p, "pid", -1))
                if pid not in target_pids:
                    continue
                mem = getattr(p, "usedGpuMemory", None)
                if isinstance(mem, int) and 0 <= mem < (1 << 60):
                    prev = per_pid_bytes.get(pid, 0)
                    if mem > prev:
                        per_pid_bytes[pid] = mem
                else:
                    self._warn_once(
                        f"vram_na_{pid}",
                        f"NVML hlásí usedGpuMemory=N/A pro PID {pid} (omezení WDDM ovladače).",
                    )
        return sum(per_pid_bytes.values()) / (1024.0 * 1024.0)

    def sample(self, target_pids: set[int]) -> dict[str, float]:
        out = {
            "gpu_pid_vram_mb": 0.0,
            "gpu_sm_util_pct": 0.0,
            "gpu_mem_ctrl_pct": 0.0,
            "pcie_tx_kbps": 0.0,
            "pcie_rx_kbps": 0.0,
            "gpu_power_w": 0.0,
            "gpu_core_mhz": 0.0,
            "gpu_mem_mhz": 0.0,
        }
        if not self.available:
            return out

        out["gpu_pid_vram_mb"] = self._query_pid_vram_mb(target_pids)

        try:
            rates = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            out["gpu_sm_util_pct"] = float(rates.gpu)
            out["gpu_mem_ctrl_pct"] = float(rates.memory)
        except Exception as exc:
            self._warn_once("util", f"NVML utilization nedostupné: {exc}")

        try:
            out["pcie_tx_kbps"] = float(
                pynvml.nvmlDeviceGetPcieThroughput(self._handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            )
            out["pcie_rx_kbps"] = float(
                pynvml.nvmlDeviceGetPcieThroughput(self._handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
            )
        except Exception as exc:
            self._warn_once("pcie", f"NVML PCIe throughput nedostupný: {exc}")

        try:
            out["gpu_power_w"] = float(pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
        except Exception as exc:
            self._warn_once("power", f"NVML power usage nedostupné: {exc}")

        try:
            out["gpu_core_mhz"] = float(pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_GRAPHICS))
            out["gpu_mem_mhz"] = float(pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_MEM))
        except Exception as exc:
            self._warn_once("clocks", f"NVML takty GPU nedostupné: {exc}")

        return out

    def close(self) -> None:
        if self.available and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception as exc:
                self._warn_once("shutdown", f"NVML shutdown varování: {exc}")
            self.available = False


def discover_project_pids() -> list[tuple[int, str]]:
    """
    Vyhledá běžící procesy projektu (launcher.py, tactical_hud.py, run_headless.py)
    a odfiltruje prázdné rodičovské Windows .venv\\Scripts\\python.exe stub procesy,
    které jen spouštějí reálný child interpret.
    """
    if psutil is None:
        return []
    own_pid = os.getpid()
    found: list[tuple[int, int, str, float]] = []  # (pid, ppid, label, create_time)
    for proc in psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time"]):
        try:
            pid = proc.info["pid"]
            if pid == own_pid:
                continue
            cmdline = proc.info.get("cmdline") or []
            joined = " ".join(cmdline).lower()
            if "hw_probe.py" in joined or "aegis_tool.py" in joined or "hub.py" in joined:
                continue
            for target in PROJECT_SCRIPTS:
                if target in joined:
                    ppid = int(proc.info.get("ppid") or 0)
                    found.append((pid, ppid, target, float(proc.info.get("create_time") or 0.0)))
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    # Odfiltrování rodičovského .venv stub procesu: pokud má nalezený proces potomka
    # se stejným cílovým skriptem, rodič je jen prázdný Windows venv wrapper.
    parent_pids_with_matching_child = {ppid for _pid, ppid, _lbl, _ct in found}
    filtered = [item for item in found if item[0] not in parent_pids_with_matching_child]
    if not filtered:
        filtered = found

    filtered.sort(key=lambda item: item[3], reverse=True)
    return [(pid, label) for pid, _ppid, label, _ct in filtered]


class ProcessTreeProbe:
    """Sleduje strom procesů (kořenové PID + potomci) pomocí psutil bez zásahu do cíle."""

    def __init__(self, root_pids: list[int]):
        if psutil is None:
            raise RuntimeError("Knihovna psutil není nainstalována.")
        self.root_pids = list(dict.fromkeys(root_pids))
        self.logical_cores = max(1, psutil.cpu_count(logical=True) or 1)
        self._procs: dict[int, psutil.Process] = {}
        self._prev_thread_times: dict[tuple[int, int], float] = {}
        self._thread_cum_delta: dict[tuple[int, int], float] = {}
        self._last_sample_ts = time.perf_counter()
        self._warned: set[str] = set()

        self._refresh_tree()
        self._prime_counters()

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            print(f"[hw_probe WARN] {msg}", file=sys.stderr)

    def _refresh_tree(self) -> list[psutil.Process]:
        active_pids: set[int] = set()
        for rpid in list(self.root_pids):
            try:
                root = self._procs.get(rpid) or psutil.Process(rpid)
                if not root.is_running() or root.status() == psutil.STATUS_ZOMBIE:
                    continue
                self._procs[rpid] = root
                active_pids.add(rpid)
                for child in root.children(recursive=True):
                    cpid = child.pid
                    if cpid not in self._procs:
                        self._procs[cpid] = child
                    active_pids.add(cpid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        for stale in list(self._procs.keys()):
            if stale not in active_pids:
                self._procs.pop(stale, None)

        return [self._procs[pid] for pid in sorted(active_pids) if pid in self._procs]

    def _prime_counters(self) -> None:
        for pid, proc in list(self._procs.items()):
            try:
                proc.cpu_percent(interval=None)
                for th in proc.threads():
                    self._prev_thread_times[(pid, th.id)] = float(th.user_time + th.system_time)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        self._last_sample_ts = time.perf_counter()

    def sample(self) -> tuple[dict[str, Any] | None, set[int]]:
        procs = self._refresh_tree()
        if not procs:
            return None, set()

        now = time.perf_counter()
        dt = max(1e-3, now - self._last_sample_ts)
        self._last_sample_ts = now

        cpu_total = 0.0
        rss_bytes = 0
        uss_bytes = 0
        private_bytes = 0
        uss_ok = False
        priv_ok = False
        num_threads = 0
        affinity_str = "n/a"
        thread_deltas: list[tuple[float, int, int, float]] = []
        live_pids: set[int] = set()

        for proc in procs:
            pid = proc.pid
            try:
                cpu_total += float(proc.cpu_percent(interval=None))
                mem = proc.memory_info()
                rss_bytes += int(mem.rss)
                num_threads += int(proc.num_threads())
                live_pids.add(pid)

                if pid == self.root_pids[0] and affinity_str == "n/a":
                    try:
                        affinity_str = _format_affinity(proc.cpu_affinity())
                    except (psutil.AccessDenied, AttributeError, OSError):
                        pass

                try:
                    full_mem = proc.memory_full_info()
                    if hasattr(full_mem, "uss"):
                        uss_bytes += int(full_mem.uss)
                        uss_ok = True
                    if hasattr(full_mem, "private"):
                        private_bytes += int(full_mem.private)
                        priv_ok = True
                except psutil.AccessDenied:
                    self._warn_once(
                        f"mem_full_{pid}",
                        f"AccessDenied při čtení memory_full_info (USS/Private) pro PID {pid}.",
                    )
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue

                try:
                    for th in proc.threads():
                        key = (pid, th.id)
                        total_t = float(th.user_time + th.system_time)
                        prev_t = self._prev_thread_times.get(key)
                        self._prev_thread_times[key] = total_t
                        if prev_t is not None:
                            delta_t = max(0.0, total_t - prev_t)
                            self._thread_cum_delta[key] = self._thread_cum_delta.get(key, 0.0) + delta_t
                            th_pct = (delta_t / dt) * 100.0
                            thread_deltas.append((th_pct, pid, th.id, total_t))
                        else:
                            thread_deltas.append((0.0, pid, th.id, total_t))
                except psutil.AccessDenied:
                    self._warn_once(f"threads_{pid}", f"AccessDenied při čtení threads() pro PID {pid}.")
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        if not live_pids:
            return None, set()

        thread_deltas.sort(key=lambda item: (item[0], item[3]), reverse=True)
        top_threads = "|".join(f"T{tid}:{pct:.1f}%" for pct, _pid, tid, _tot in thread_deltas[:3])
        if not top_threads:
            top_threads = "-"

        return (
            {
                "root_pids": "+".join(str(p) for p in self.root_pids if p in live_pids),
                "proc_count": len(live_pids),
                "cpu_total_pct": cpu_total,
                "cpu_norm_pct": cpu_total / self.logical_cores,
                "num_threads": num_threads,
                "affinity": affinity_str,
                "top_threads": top_threads,
                "ram_rss_mb": rss_bytes / (1024.0 * 1024.0),
                "ram_uss_mb": (uss_bytes / (1024.0 * 1024.0)) if uss_ok else float("nan"),
                "ram_private_mb": (private_bytes / (1024.0 * 1024.0)) if priv_ok else float("nan"),
            },
            live_pids,
        )

    def get_top_cumulative_threads(self, limit: int = 5) -> list[tuple[int, int, float]]:
        items = [(pid, tid, sec) for (pid, tid), sec in self._thread_cum_delta.items() if sec > 0.0]
        items.sort(key=lambda x: x[2], reverse=True)
        return items[:limit]


def _diagnose_bottlenecks(samples: list[dict[str, Any]]) -> list[str]:
    if not samples:
        return ["Nedostatek vzorků pro diagnózu."]

    _, cpu_avg, cpu_pk, _ = _compute_stats([float(s["cpu_total_pct"]) for s in samples])
    _, sm_avg, _, _ = _compute_stats([float(s["gpu_sm_util_pct"]) for s in samples])
    _, rx_avg_mb, rx_pk_mb, _ = _compute_stats([float(s["pcie_rx_kbps"]) / 1024.0 for s in samples])
    _, tx_avg_mb, tx_pk_mb, _ = _compute_stats([float(s["pcie_tx_kbps"]) / 1024.0 for s in samples])
    rss_vals = [float(s["ram_rss_mb"]) for s in samples]
    rss_growth = (rss_vals[-1] - rss_vals[0]) if len(rss_vals) >= 2 else 0.0

    findings: list[str] = []
    if cpu_avg >= 85.0 and sm_avg < 35.0:
        findings.append(
            f"**CPU / GIL bottleneck:** Proces vytěžuje ~{cpu_avg:.1f} % jednoho jádra (peak {cpu_pk:.1f} %), "
            f"zatímco GPU SM je vytíženo jen na {sm_avg:.1f} %. Brzdou je CPU dispatch nebo synchronizace."
        )
    elif sm_avg >= 80.0:
        findings.append(f"**GPU-bound:** GPU SM je silně vytíženo (průměr {sm_avg:.1f} %).")

    if rx_avg_mb >= 500.0 or tx_avg_mb >= 500.0:
        findings.append(
            f"**Vysoký PCIe provoz:** Průměrně RX {rx_avg_mb:.1f} MB/s (peak {rx_pk_mb:.1f} MB/s), "
            f"TX {tx_avg_mb:.1f} MB/s (peak {tx_pk_mb:.1f} MB/s) – zkontrolujte časté GPU↔CPU přenosy."
        )

    if rss_growth > 100.0:
        findings.append(f"**Růst RAM během měření:** RSS vzrostla o +{rss_growth:.1f} MB.")

    if not findings:
        findings.append(
            f"Žádné kritické úzké hrdlo nebylo detekováno (CPU avg {cpu_avg:.1f} %, GPU SM avg {sm_avg:.1f} %)."
        )
    return findings


def _write_markdown_report(
    report_path: Path,
    csv_path: Path,
    target_desc: str,
    gpu_name: str,
    interval_ms: int,
    elapsed_total: float,
    tag: str,
    samples: list[dict[str, Any]],
    top_cum_threads: list[tuple[int, int, float]],
) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        f"# HW Probe Report{f' – `{tag}`' if tag else ''}",
        "",
        "## Metadata běhu",
        f"- **Datum a čas:** {now_str}",
        f"- **Cílový proces:** `{target_desc}`",
        f"- **GPU:** {gpu_name}",
        f"- **Interval / Délka:** {interval_ms} ms / {elapsed_total:.1f} s ({len(samples)} vzorků)",
        f"- **Surová data (CSV):** `{csv_path.name}`",
        "",
        "## Souhrn metrik (MIN / AVG / PEAK / P95)",
        "",
        "| Metrika | MIN | AVG | PEAK | P95 |",
        "|---|---:|---:|---:|---:|",
    ]

    table_spec = [
        ("CPU strom (%)", lambda s: float(s["cpu_total_pct"])),
        ("RAM RSS (MB)", lambda s: float(s["ram_rss_mb"])),
        ("RAM USS (MB)", lambda s: float(s["ram_uss_mb"])),
        ("GPU PID VRAM (MB)", lambda s: float(s["gpu_pid_vram_mb"])),
        ("GPU SM (%)", lambda s: float(s["gpu_sm_util_pct"])),
        ("GPU MC (%)", lambda s: float(s["gpu_mem_ctrl_pct"])),
        ("PCIe RX (MB/s)", lambda s: float(s["pcie_rx_kbps"]) / 1024.0),
        ("PCIe TX (MB/s)", lambda s: float(s["pcie_tx_kbps"]) / 1024.0),
        ("GPU Power (W)", lambda s: float(s["gpu_power_w"])),
    ]

    for label, extractor in table_spec:
        vals = [extractor(s) for s in samples]
        valid = [v for v in vals if not math.isnan(v)]
        if not valid:
            lines.append(f"| {label} | n/a | n/a | n/a | n/a |")
        else:
            mn, avg, pk, p95 = _compute_stats(valid)
            lines.append(f"| {label} | {mn:.1f} | {avg:.1f} | {pk:.1f} | {p95:.1f} |")

    lines.extend(["", "## Top vlákna (kumulativní CPU čas)", ""])
    if top_cum_threads:
        for pid, tid, sec in top_cum_threads:
            lines.append(f"- `PID {pid} / T{tid}`: **{sec:.2f} s**")
    else:
        lines.append("- *(nebyla naměřena aktivita vláken)*")

    lines.extend(["", "## Automatická diagnóza", ""])
    for finding in _diagnose_bottlenecks(samples):
        lines.append(f"- {finding}")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _print_summary(
    samples: list[dict[str, Any]],
    csv_path: Path | None,
    report_path: Path | None,
    top_cum_threads: list[tuple[int, int, float]],
) -> None:
    print("\n" + "=" * 86)
    log_info = f"CSV={csv_path.name} | MD={report_path.name}" if (csv_path and report_path) else "logování vypnuto (--no-log)"
    print(f"[HW PROBE SUMMARY] vzorků={len(samples)} | {log_info}")
    print("-" * 86)
    if not samples:
        print("  Žádné vzorky nebyly nasbírány.")
        print("=" * 86)
        return

    metrics = [
        ("CPU strom (%)", lambda s: float(s["cpu_total_pct"])),
        ("RAM RSS (MB)", lambda s: float(s["ram_rss_mb"])),
        ("RAM USS (MB)", lambda s: float(s["ram_uss_mb"])),
        ("GPU PID VRAM (MB)", lambda s: float(s["gpu_pid_vram_mb"])),
        ("GPU SM (%)", lambda s: float(s["gpu_sm_util_pct"])),
        ("PCIe RX (MB/s)", lambda s: float(s["pcie_rx_kbps"]) / 1024.0),
        ("PCIe TX (MB/s)", lambda s: float(s["pcie_tx_kbps"]) / 1024.0),
        ("GPU Power (W)", lambda s: float(s["gpu_power_w"])),
    ]
    print(f"  {'Metrika':<20} {'MIN':>10} {'AVG':>10} {'PEAK':>10} {'P95':>10}")
    print("  " + "-" * 64)
    for label, extractor in metrics:
        vals = [extractor(s) for s in samples]
        valid = [v for v in vals if not math.isnan(v)]
        if not valid:
            print(f"  {label:<20} {'n/a':>10} {'n/a':>10} {'n/a':>10} {'n/a':>10}")
            continue
        mn, avg, pk, p95 = _compute_stats(valid)
        print(f"  {label:<20} {mn:10.1f} {avg:10.1f} {pk:10.1f} {p95:10.1f}")

    if top_cum_threads:
        print("-" * 86)
        th_str = ", ".join(f"PID {pid}/T{tid}: {sec:.2f}s" for pid, tid, sec in top_cum_threads)
        print(f"  Top vlákna (CPU čas): {th_str}")
    print("=" * 86)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hw_probe",
        description="Neinvazivní per-process hardwarová sonda pro Cyber-Tactics 2.0 (psutil + NVML).",
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--pid", type=int, help="PID cílového procesu (sleduje i jeho child procesy).")
    mode.add_argument(
        "--auto",
        action="store_true",
        help="Automaticky vyhledá běžící proces projektu (výchozí režim při spuštění bez parametrů).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=250,
        help="Vzorkovací interval v milisekundách (výchozí: 250 ms).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Volitelný časový limit běhu v sekundách (bez něj běží do Ctrl+C).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Volitelný krátký popisek běhu přidaný do názvů výstupních souborů.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Vypne zápis CSV a Markdown reportu (pouze výpis do konzole).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Index NVIDIA GPU pro NVML (výchozí: 0).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Cílová složka pro výstupní logy (výchozí: devtools/logs/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = parse_args(argv)

    if psutil is None:
        print("[hw_probe ERROR] Knihovna psutil není nainstalována.", file=sys.stderr)
        return 2

    if args.interval < 50:
        print("[hw_probe WARN] Interval < 50 ms upraven na 50 ms pro minimální režii.", file=sys.stderr)
        args.interval = 50
    interval_s = args.interval / 1000.0

    use_auto = args.auto or (args.pid is None)
    if use_auto:
        discovered = discover_project_pids()
        if not discovered:
            print(
                "[hw_probe] Nebyl nalezen žádný běžící proces projektu "
                f"({', '.join(PROJECT_SCRIPTS)}).",
                file=sys.stderr,
            )
            return 1
        root_pids = [pid for pid, _ in discovered]
        desc = ", ".join(f"{label} (PID {pid})" for pid, label in discovered)
    else:
        if not psutil.pid_exists(args.pid):
            print(f"[hw_probe ERROR] Proces s PID {args.pid} neexistuje.", file=sys.stderr)
            return 1
        root_pids = [args.pid]
        desc = f"PID {args.pid}"

    tag = _sanitize_tag(args.tag)
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"hw_probe_{ts_tag}_{tag}" if tag else f"hw_probe_{ts_tag}"

    csv_path: Path | None = None
    report_path: Path | None = None
    if not args.no_log:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{base_name}.csv"
        report_path = out_dir / f"{base_name}_report.md"

    stop_requested = False

    def _handle_sig(_sig: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_sig)

    proc_probe = ProcessTreeProbe(root_pids)
    nvml_probe = NVMLProbe(gpu_index=args.gpu)

    print(
        f"[HW PROBE] Cíl: {desc} | GPU: {nvml_probe.gpu_name} | "
        f"Interval: {args.interval} ms | Limit: {f'{args.duration:.1f}s' if args.duration else 'Ctrl+C'}"
    )
    if csv_path and report_path:
        print(f"[HW PROBE] Výstup: {csv_path} (+ {report_path.name})")
    else:
        print("[HW PROBE] Logování do souborů vypnuto (--no-log).")
    print(
        f"{'T(s)':>6} | {'CPU%':>6} {'THR':>3} | {'RSS':>7} {'USS':>7} | "
        f"{'pVRAM':>7} {'SM%':>4} {'MC%':>4} {'PWR':>6} {'CLK':>9} | {'PCIe RX/TX MB/s':>17} | TOP VLÁKNA"
    )
    print("-" * 114)

    all_samples: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    start_ts = time.perf_counter()

    def _flush_csv(writer: csv.DictWriter | None, fh: Any) -> None:
        if writer is None or fh is None or not batch_rows:
            return
        writer.writerows(batch_rows)
        fh.flush()
        batch_rows.clear()

    fh: Any = None
    writer: csv.DictWriter | None = None
    try:
        if csv_path is not None:
            fh = open(csv_path, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()

        next_tick = start_ts + interval_s
        while not stop_requested:
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.1))
                if time.perf_counter() < next_tick:
                    continue
            next_tick = time.perf_counter() + interval_s

            elapsed = time.perf_counter() - start_ts
            proc_data, live_pids = proc_probe.sample()
            if proc_data is None:
                print("\n[HW PROBE] Sledovaný proces byl ukončen.")
                break

            gpu_data = nvml_probe.sample(live_pids)
            row: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "elapsed_s": round(elapsed, 3),
                **{
                    k: (round(v, 2) if isinstance(v, float) and not math.isnan(v) else ("" if isinstance(v, float) else v))
                    for k, v in proc_data.items()
                },
                **{k: round(v, 2) for k, v in gpu_data.items()},
            }
            batch_rows.append(row)
            all_samples.append({**proc_data, **gpu_data})

            if len(batch_rows) >= BATCH_FLUSH_SIZE:
                _flush_csv(writer, fh)

            uss_txt = f"{proc_data['ram_uss_mb']:6.1f}M" if not math.isnan(proc_data["ram_uss_mb"]) else "    n/a"
            clk_txt = f"{int(gpu_data['gpu_core_mhz'])}/{int(gpu_data['gpu_mem_mhz'])}"
            pcie_txt = f"{gpu_data['pcie_rx_kbps'] / 1024.0:7.1f}/{gpu_data['pcie_tx_kbps'] / 1024.0:7.1f}"
            print(
                f"{elapsed:6.1f} | {proc_data['cpu_total_pct']:5.1f}% {proc_data['num_threads']:3d} | "
                f"{proc_data['ram_rss_mb']:6.1f}M {uss_txt} | "
                f"{gpu_data['gpu_pid_vram_mb']:6.1f}M {gpu_data['gpu_sm_util_pct']:3.0f}% {gpu_data['gpu_mem_ctrl_pct']:3.0f}% "
                f"{gpu_data['gpu_power_w']:5.1f}W {clk_txt:>9} | {pcie_txt:>17} | {proc_data['top_threads']}"
            )

            if args.duration is not None and elapsed >= args.duration:
                break

        _flush_csv(writer, fh)
    except KeyboardInterrupt:
        pass
    finally:
        if fh is not None:
            try:
                _flush_csv(writer, fh)
                fh.close()
            except OSError:
                pass
        elapsed_total = time.perf_counter() - start_ts
        top_threads = proc_probe.get_top_cumulative_threads(limit=5)
        if report_path is not None and csv_path is not None:
            try:
                _write_markdown_report(
                    report_path=report_path,
                    csv_path=csv_path,
                    target_desc=desc,
                    gpu_name=nvml_probe.gpu_name,
                    interval_ms=args.interval,
                    elapsed_total=elapsed_total,
                    tag=tag,
                    samples=all_samples,
                    top_cum_threads=top_threads,
                )
            except OSError as exc:
                print(f"[hw_probe WARN] Nepodařilo se zapsat Markdown report: {exc}", file=sys.stderr)
        nvml_probe.close()
        _print_summary(all_samples, csv_path, report_path, top_threads)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
