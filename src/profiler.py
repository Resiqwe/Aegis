"""
profiler.py - Lehký diagnostický stopkový měřič pro přesnou identifikaci lagů.
Nemá prakticky žádný overhead (sbírá jen time.perf_counter).
Každé 2 sekundy vypíše agregované metriky do konzole.
"""

import time
from collections import defaultdict

class MetricAccumulator:
    def __init__(self, name: str, interval_sec: float = 2.0):
        self.name = name
        self.interval_sec = interval_sec
        self.last_report = time.perf_counter()
        self.counts = defaultdict(int)
        self.sums = defaultdict(float)
        self.maxs = defaultdict(float)

    def record(self, key: str, duration_ms: float):
        self.counts[key] += 1
        self.sums[key] += duration_ms
        if duration_ms > self.maxs[key]:
            self.maxs[key] = duration_ms

    def should_report(self) -> bool:
        return (time.perf_counter() - self.last_report) >= self.interval_sec

    def report_and_reset(self, extra_header: str = ""):
        now = time.perf_counter()
        dur = max(0.001, now - self.last_report)
        print(f"\n===== [PERF DIAGNOSTIC: {self.name}] (last {dur:.1f}s) {extra_header} =====")
        for key in sorted(self.counts.keys()):
            cnt = self.counts[key]
            avg_ms = self.sums[key] / max(1, cnt)
            max_ms = self.maxs[key]
            print(f"  {key:<18}: avg={avg_ms:6.2f} ms | max={max_ms:6.2f} ms | calls={cnt}")
        print("=" * 60)
        self.last_report = now
        self.counts.clear()
        self.sums.clear()
        self.maxs.clear()

# Globální instance pro Worker a pro Render
worker_prof = MetricAccumulator("SimWorker Thread", interval_sec=2.0)
render_prof = MetricAccumulator("Render / UI Thread", interval_sec=2.0)
