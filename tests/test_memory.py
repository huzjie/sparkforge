"""Unified-memory scheduler / tiers."""
import unittest

from sparkforge.memory.scheduler import UnifiedMemoryScheduler
from sparkforge.memory.tiers import MemoryTiers
from sparkforge.memory.prefetch import PrefetchPipeline
from sparkforge.memory.bandwidth import tokens_per_second, quantized_footprint_gb


class TestMemory(unittest.TestCase):
    def test_scheduler_prefetch_reduces_stall(self):
        sched = UnifiedMemoryScheduler(800.0, prefetch_depth=8)
        resident = set(range(4))
        top = [0, 1, 2, 3, 4, 5, 6, 7]
        # no prefetch: 4 misses stall
        s1 = sched.step(top, resident, 32, 64)
        self.assertGreater(s1, 0.0)
        # prefetch next: 4 hits
        sched.prefetch([4, 5, 6, 7], resident, 32, 64)
        s2 = sched.step([4, 5, 6, 7], resident, 32, 64)
        self.assertLess(s2, s1)

    def test_tiers_place(self):
        tiers = MemoryTiers(1.0, 10.0, 100.0)
        t = tiers.place(0, 32, 64, heat=1.0)
        self.assertIn(t.value, ("gpu", "unified", "ssd"))

    def test_prefetch_pipeline(self):
        pp = PrefetchPipeline(800.0, 8)
        c, t = pp.run([0, 1, 2], [4, 5, 6, 7, 8, 9], 32, 64)
        self.assertGreaterEqual(c, 0.0)
        self.assertGreater(t, 0.0)

    def test_quantized_footprint(self):
        # 816B at 4-bit ~ 381 GB (less than 512GB unified pool)
        gb = quantized_footprint_gb(816.0, 4)
        self.assertLess(gb, 500.0)

    def test_tps_scales_with_hitrate(self):
        low = tokens_per_second(800.0, 84e9, 0.0)
        high = tokens_per_second(800.0, 84e9, 0.99)
        self.assertGreater(high, low)


if __name__ == "__main__":
    unittest.main()
