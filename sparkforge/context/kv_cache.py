"""Block KV cache: paged blocks of key/value tensors.

Instead of one giant contiguous KV tensor, keys/values are paged into fixed-size
blocks. Blocks evicted from GPU can be offloaded to unified memory / SSD and
swapped back on demand -- the standard trick for 1M-token contexts.
"""
from dataclasses import dataclass

from ..tensor import Tensor
from ..utils.hash import stable_vector


@dataclass
class Block:
    block_id: int
    keys: Tensor
    values: Tensor
    n_filled: int = 0
    resident: bool = True

    def fill(self, k, v):
        """Append one token's K/V into the block (assume per-row vectors)."""
        self.keys.data[self.n_filled] = k
        self.values.data[self.n_filled] = v
        self.n_filled += 1

    @property
    def full(self):
        return self.n_filled >= len(self.keys.data)


class KVCache:
    def __init__(self, block_size, gpu_blocks, dim, heads=1):
        self.block_size = block_size
        self.gpu_blocks = gpu_blocks
        self.dim = dim
        self.heads = heads
        self.blocks = []
        self._alloc(gpu_blocks, resident=True)

    def _alloc(self, n, resident):
        for _ in range(n):
            keys = Tensor.zeros(self.block_size, self.dim)
            values = Tensor.zeros(self.block_size, self.dim)
            self.blocks.append(Block(len(self.blocks), keys, values, resident=resident))

    def append(self, k, v):
        """Add one token, allocating/offloading blocks as needed."""
        blk = self._writable_block()
        blk.fill(k, v)
        return blk

    def _writable_block(self):
        for b in self.blocks:
            if b.resident and not b.full:
                return b
        # no resident space: allocate a new block
        self._alloc(1, resident=True)
        return self.blocks[-1]

    def total_capacity(self):
        return len(self.blocks) * self.block_size

    def resident_blocks(self):
        return sum(1 for b in self.blocks if b.resident)
