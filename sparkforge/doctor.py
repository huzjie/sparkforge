"""Self-check: instantiate backends, run forward, verify components."""
import sys

from .config import load_config


def main():
    cfg = load_config("configs/default.yaml")
    from .backends import list_backends, get_backend
    print("[doctor] backends:", list_backends())
    for name in list_backends():
        b = get_backend(name, cfg)
        out = b.generate("hello world", max_tokens=4)
        print("[doctor] %-14s -> %s" % (name, out[:40]))

    from .model.spark import SparkModel
    model = SparkModel(cfg)
    out, aux, activated = model.forward("def add(a, b): return a + b")
    print("[doctor] forward out dim=%d aux=%.4f activated_layers=%d" % (
        len(out.data[0]), aux, len(activated)))

    from .moe.layer import MoELayer
    layer = MoELayer(cfg.moe.n_experts, cfg.moe.top_k, cfg.moe.dim, cfg.moe.hidden)
    ratio = layer.active_ratio()
    print("[doctor] MoE active_ratio(top_k/n_experts)=%.4f  declared=%dB/%dB active" % (
        ratio, cfg.moe.total_params_b, cfg.moe.active_params_b))

    from .memory.scheduler import UnifiedMemoryScheduler
    sched = UnifiedMemoryScheduler(cfg.memory.bandwidth_gbps, cfg.memory.prefetch_depth)
    print("[doctor] memory scheduler bandwidth=%.1f GB/s" % sched.bandwidth_gbps)

    from .context.block_manager import BlockManager
    bm = BlockManager(cfg.context.block_size, cfg.context.gpu_blocks, cfg.model.dim)
    print("[doctor] context block_size=%d gpu_blocks=%d" % (cfg.context.block_size, cfg.context.gpu_blocks))

    from .bench.summarize import run_all
    report = run_all(cfg)
    print("[doctor] bench AVG=%.4f" % report["avg"])
    print("[doctor] ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
