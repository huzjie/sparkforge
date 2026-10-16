"""Bandwidth-aware expert scheduler for unified memory.

The scheduling insight behind "run 816B on two Mac Studios": the M5 Ultra's
integrated GPU streams expert weights from unified memory at ~1.2 TB/s, which is
enough to read the small active slice (42B) per token. We model latency as
``bytes / bandwidth`` and demonstrate that prefetching hides it behind compute.
"""
from ..utils.hash import stable_float


class UnifiedMemoryScheduler:
    def __init__(self, bandwidth_gbps, prefetch_depth=8):
        self.bandwidth_gbps = bandwidth_gbps  # GB/s
        self.prefetch_depth = prefetch_depth
        self.total_bytes_moved = 0.0
        self.total_stalls = 0.0
        self.prefetch_hits = 0
        self.prefetch_total = 0
        self._prefetched = set()

    def bytes_to_read(self, expert_id, dim, hidden):
        """2 bytes fp16 * 3 matrices."""
        return 3 * dim * hidden * 2

    def transfer_seconds(self, n_bytes):
        return n_bytes / (self.bandwidth_gbps * 1e9)

    def step(self, top_k_experts, resident, dim, hidden):
        """Simulate one token's expert reads; return wall-clock seconds.

        ``resident`` is the set of experts already in the fast pool. Experts not
        resident incur a stall (unless prefetched in a prior step).
        """
        stall = 0.0
        for eid in top_k_experts:
            self.prefetch_total += 1
            if eid in resident or eid in self._prefetched:
                self.prefetch_hits += 1
                continue
            n_bytes = self.bytes_to_read(eid, dim, hidden)
            stall += self.transfer_seconds(n_bytes)
            self.total_bytes_moved += n_bytes
        self.total_stalls += stall
        return stall

    def prefetch(self, next_top_k, resident, dim, hidden):
        """Prefetch next step's experts into a shadow fast set."""
        self._prefetched = set()
        for eid in next_top_k[: self.prefetch_depth]:
            if eid not in resident:
                self._prefetched.add(eid)
                self.total_bytes_moved += self.bytes_to_read(eid, dim, hidden)
