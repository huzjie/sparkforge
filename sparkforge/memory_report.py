"""Standalone memory-tier report (helpful for capacity planning)."""
from .config import load_config
from .memory.tiers import MemoryTiers
from .memory.bandwidth import memory_footprint_gb, quantized_footprint_gb


def main():
    cfg = load_config("configs/default.yaml")
    total = cfg.moe.total_params_b
    print("Muse Spark Open (declared) scale:")
    print("  total params       : %.0f B" % total)
    print("  active per token   : %.0f B" % cfg.moe.active_params_b)
    print("  fp16 footprint     : %.1f GB" % memory_footprint_gb(total, 16))
    print("  nvfp4 footprint    : %.1f GB" % quantized_footprint_gb(total, 4))
    print("  experts/layer      : %d (top-%d)" % (cfg.moe.n_experts, cfg.moe.top_k))
    m = cfg.memory
    print("Unified-memory profile:")
    print("  GPU pool           : %.0f GB" % m.gpu_pool_gb)
    print("  unified pool       : %.0f GB" % m.unified_pool_gb)
    print("  SSD pool           : %.0f GB" % m.ssd_pool_gb)
    print("  bandwidth          : %.0f GB/s" % m.bandwidth_gbps)


if __name__ == "__main__":
    main()
