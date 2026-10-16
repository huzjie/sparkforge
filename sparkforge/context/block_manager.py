"""Offload manager: move cold KV blocks to unified memory, swap hot ones back.

Tracks access recency per block; when resident blocks exceed the GPU budget, the
least-recently-used resident block is offloaded (marked non-resident) and, on a
cache hit to an offloaded block, it is swapped back in (possibly evicting
another). This is the 1M-context enabler.
"""
from collections import OrderedDict

from .kv_cache import KVCache


class BlockManager:
    def __init__(self, block_size, gpu_blocks, dim, offload=True):
        self.cache = KVCache(block_size, gpu_blocks, dim)
        self.offload = offload
        self.recency = OrderedDict()
        self.offload_count = 0
        self.swap_count = 0

    def append(self, k, v):
        blk = self.cache.append(k, v)
        self._touch(blk.block_id)
        self._enforce_budget()
        return blk

    def _touch(self, block_id):
        self.recency[block_id] = True
        self.recency.move_to_end(block_id)

    def _enforce_budget(self):
        if not self.offload:
            return
        budget = self.cache.gpu_blocks
        resident = [b for b in self.cache.blocks if b.resident]
        overflow = len(resident) - budget
        while overflow > 0:
            # evict LRU resident block
            for bid in self.recency:
                blk = self.cache.blocks[bid]
                if blk.resident:
                    blk.resident = False
                    self.offload_count += 1
                    break
            overflow -= 1

    def lookup(self, block_id):
        """Access a block; if it was offloaded, swap it back in."""
        blk = self.cache.blocks[block_id]
        if not blk.resident:
            self._enforce_budget()
            blk.resident = True
            self.swap_count += 1
        self._touch(block_id)
        return blk

    def stats(self):
        return {
            "offload_count": self.offload_count,
            "swap_count": self.swap_count,
            "resident_blocks": self.cache.resident_blocks(),
            "total_blocks": len(self.cache.blocks),
        }
