"""Bandwidth benchmark: tokens/sec as a function of cache-hit rate."""
from .registry import register_benchmark
from ..memory.bandwidth import tokens_per_second, memory_footprint_gb, quantized_footprint_gb


@register_benchmark("bandwidth")
def run(cfg, backend=None):
    bw = cfg.memory.bandwidth_gbps
    # active slice 42B fp16 -> ~84 GB; realistic systems stream only misses
    active_bytes = cfg.moe.active_params_b * 1e9 * 2  # fp16
    hit_rate = 0.99
    tps = tokens_per_second(bw, active_bytes, hit_rate)
    # normalized score: higher is better; cap at a sane ceiling
    score = min(1.0, tps / 100.0)
    return score
