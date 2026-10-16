"""Routing recall evaluation over a token stream."""
from ..oracle import gold_experts


def routing_recall(backend, n_tokens, top_k, n_experts, n_layers):
    hits = 0
    total = 0
    for t in range(n_tokens):
        for layer in range(n_layers):
            gold = set(gold_experts("tok-%d" % t, layer, top_k, n_experts))
            chosen = set(backend.route("tok-%d" % t, layer, top_k))
            hits += len(gold & chosen)
            total += top_k
    return hits / total
