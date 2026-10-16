"""Bandwidth budget model for a dual-Mac-Studio setup.

Helps answer "how many tokens per second can I get at what bandwidth", given the
active-parameter slice that must be streamed per token. Reference: M5 Ultra ~
1.2 TB/s unified-memory bandwidth; streaming 42B fp16 (~84 GB) per token would be
impractical, so real systems cache the hottest experts and stream only misses --
we model that hit/miss trade-off.
"""


def tokens_per_second(bandwidth_gbps, active_bytes_per_token, cache_hit_rate):
    """Streaming throughput: miss bytes must come from memory, hits are free."""
    miss_bytes = active_bytes_per_token * (1.0 - cache_hit_rate)
    if miss_bytes <= 0:
        return float("inf")
    return (bandwidth_gbps * 1e9) / miss_bytes


def memory_footprint_gb(total_params_b, bits_per_param=16):
    """Raw weight footprint in GB for a given bit width."""
    return total_params_b * 1e9 * bits_per_param / 8 / (1024 ** 3)


def quantized_footprint_gb(total_params_b, bits_per_param=4):
    return memory_footprint_gb(total_params_b, bits_per_param)
