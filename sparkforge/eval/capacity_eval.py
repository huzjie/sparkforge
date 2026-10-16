"""Capacity report (footprint vs unified-memory pool)."""
from ..memory.bandwidth import memory_footprint_gb, quantized_footprint_gb


def capacity_report(total_params_b, unified_pool_gb):
    return {
        "total_params_b": total_params_b,
        "fp16_gb": memory_footprint_gb(total_params_b, 16),
        "nvfp4_gb": quantized_footprint_gb(total_params_b, 4),
        "fits_unified": quantized_footprint_gb(total_params_b, 4) <= unified_pool_gb,
    }
