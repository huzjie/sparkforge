"""Lookahead prefetch pipeline: hide SSD/CPU latency behind compute.

Predicts the next layer's top-k experts (using the previous layer's routing) and
issues the weight streams before they are needed. The closed-form speedup for a
compute time ``T_c`` and prefetchable transfer ``T_p`` is ``(T_c + T_p) / T_c``
when prefetch fully overlaps; we report the *effective* overlap.
"""
from ..utils.hash import stable_float


class PrefetchPipeline:
    def __init__(self, bandwidth_gbps, prefetch_depth=8):
        self.bandwidth_gbps = bandwidth_gbps
        self.prefetch_depth = prefetch_depth
        self.overlapped_bytes = 0.0
        self.total_bytes = 0.0

    def transfer_seconds(self, n_bytes):
        return n_bytes / (self.bandwidth_gbps * 1e9)

    def run(self, layer_activated, next_predicted, dim, hidden):
        """layer_activated: this layer's top-k; next_predicted: predicted next.

        Returns (compute_seconds, hidden_transfer_seconds). The bytes of the
        predicted experts are streamed during compute; overlap is the min of
        compute time and transfer time.
        """
        per_expert = 3 * dim * hidden * 2
        n_next = min(len(next_predicted), self.prefetch_depth)
        transfer = self.transfer_seconds(per_expert * n_next)
        compute = self.transfer_seconds(per_expert * len(layer_activated)) * 0.5
        overlap = min(compute, transfer)
        self.overlapped_bytes += per_expert * n_next * (overlap / transfer if transfer else 0.0)
        self.total_bytes += per_expert * n_next
        return compute, transfer
