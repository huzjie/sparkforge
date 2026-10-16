"""Routing benchmark: recall@top-k of gold experts across layers.

At skill=0.5 the mock router is random -> recall ~ top_k/n_experts. At skill=1.0
it always hits gold -> recall 1.0. This is the sparse-MoE "router quality" curve.
"""
from .registry import register_benchmark
from ..backends import get_backend


@register_benchmark("routing")
def run(cfg, backend=None, n_tokens=24):
    if backend is None:
        backend = get_backend("mock", cfg)
    top_k = cfg.moe.top_k
    n_experts = cfg.moe.n_experts
    layers = cfg.moe.n_layers
    hits = 0
    total = 0
    from ..oracle import gold_experts
    for t in range(n_tokens):
        for layer in range(layers):
            gold = set(gold_experts("tok-%d" % t, layer, top_k, n_experts))
            chosen = set(backend.route("tok-%d" % t, layer, top_k))
            hits += len(gold & chosen)
            total += top_k
    return hits / total
