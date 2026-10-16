"""KV cache + block manager offload."""
import unittest

from sparkforge.context.kv_cache import KVCache
from sparkforge.context.block_manager import BlockManager


class TestContext(unittest.TestCase):
    def test_kv_cache_append(self):
        cache = KVCache(block_size=4, gpu_blocks=2, dim=8)
        for i in range(9):
            cache.append([0.0] * 8, [0.0] * 8)
        self.assertGreaterEqual(cache.total_capacity(), 8)

    def test_block_manager_offload(self):
        bm = BlockManager(block_size=4, gpu_blocks=2, dim=8, offload=True)
        for i in range(20):
            bm.append([0.0] * 8, [0.0] * 8)
        stats = bm.stats()
        self.assertGreater(stats["offload_count"], 0)
        self.assertLessEqual(stats["resident_blocks"], stats["total_blocks"])

    def test_swap_back(self):
        bm = BlockManager(block_size=4, gpu_blocks=2, dim=8, offload=True)
        for i in range(20):
            bm.append([0.0] * 8, [0.0] * 8)
        # access an offloaded block -> swap
        for bid in range(len(bm.cache.blocks)):
            if not bm.cache.blocks[bid].resident:
                bm.lookup(bid)
                break
        self.assertGreater(bm.stats()["swap_count"], 0)


if __name__ == "__main__":
    unittest.main()
