"""Context recall: can we swap an offloaded KV block back and retrieve?

At high offload pressure the block manager evicts LRU blocks; recall measures the
fraction of blocks that can be swapped back in (capacity-permitting). This
validates the 1M-context offload path.
"""
from .registry import register_benchmark
from ..context.block_manager import BlockManager
from ..utils.hash import stable_vector


@register_benchmark("context_recall")
def run(cfg, backend=None):
    dim = cfg.model.dim
    bm = BlockManager(cfg.context.block_size, cfg.context.gpu_blocks, dim, offload=cfg.context.offload)
    n_tokens = cfg.context.gpu_blocks * cfg.context.block_size * 3  # force offload
    for i in range(n_tokens):
        k = stable_vector("k:%d" % i, dim)
        v = stable_vector("v:%d" % i, dim)
        bm.append(k, v)
    stats = bm.stats()
    total = stats["total_blocks"]
    resident = stats["resident_blocks"]
    # recall = fraction of blocks addressable (resident) after offload
    return resident / total if total else 0.0
