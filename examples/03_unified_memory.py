"""Demonstrate unified-memory scheduling + prefetch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkforge.memory.scheduler import UnifiedMemoryScheduler

sched = UnifiedMemoryScheduler(bandwidth_gbps=1200.0, prefetch_depth=8)
resident = set(range(8))
top = [0, 1, 2, 3, 8, 9, 10, 11]
cold = sched.step(top, resident, 32, 64)
sched.prefetch([8, 9, 10, 11], resident, 32, 64)
warm = sched.step([8, 9, 10, 11], resident, 32, 64)
print("cold stall: %.2e s" % cold)
print("warm stall: %.2e s (prefetched)" % warm)
