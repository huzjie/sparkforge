"""Expert tiering across a unified memory pool.

The MLX unified-memory model: CPU and GPU share *one* addressable pool, so there
are no host<->device copies -- the cost of "moving" an expert is just the memory
bandwidth to stream its weights. We model three tiers:
  * GPU (resident, fast) -- the hottest experts
  * UNIFIED (addressable, medium) -- the bulk of the 816B weights
  * SSD (cold, slow) -- overflow / never-hot experts
"""
from enum import Enum

from ..utils.hash import stable_float


class Tier(Enum):
    GPU = "gpu"
    UNIFIED = "unified"
    SSD = "ssd"


class MemoryTiers:
    def __init__(self, gpu_pool_gb, unified_pool_gb, ssd_pool_gb):
        self.pools = {Tier.GPU: gpu_pool_gb, Tier.UNIFIED: unified_pool_gb, Tier.SSD: ssd_pool_gb}
        self.used = {Tier.GPU: 0.0, Tier.UNIFIED: 0.0, Tier.SSD: 0.0}

    def expert_size_gb(self, expert_id, dim, hidden):
        """3 matrices (hidden x dim, hidden x dim, dim x hidden) * 2 bytes fp16."""
        params = 3 * dim * hidden
        return params * 2 / (1024 ** 3)

    def place(self, expert_id, dim, hidden, heat):
        """Place an expert in the hottest tier that has room (heat desc)."""
        size = self.expert_size_gb(expert_id, dim, hidden)
        for tier in (Tier.GPU, Tier.UNIFIED, Tier.SSD):
            if self.used[tier] + size <= self.pools[tier]:
                self.used[tier] += size
                return tier
        # pools exhausted: spill to SSD regardless
        self.used[Tier.SSD] += size
        return Tier.SSD

    def reset(self):
        for t in self.used:
            self.used[t] = 0.0
