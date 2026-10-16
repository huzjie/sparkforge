"""Trainable deterministic mock backend.

The mock's ``skill`` in [0,1] drives a routing signal: skill=0.5 -> the router is
random (recall = top_k/n_experts baseline), skill=1.0 -> the router always hits
the gold experts. The trainer raises skill monotonically, so every benchmark
moves cleanly from baseline to ceiling.
"""
from .registry import register_backend
from ..utils.hash import stable_float, stable_bool
from ..oracle import gold_experts


@register_backend("mock")
class MockBackend:
    def __init__(self, config):
        self.config = config
        self.skill = config.model.skill
        self.dim = config.model.dim

    def route(self, token_key, layer_id, top_k):
        """Route one token at a layer; skill blends gold experts with noise."""
        gold = gold_experts(token_key, layer_id, top_k)
        boost = (self.skill - 0.5) * 2.0  # skill=0.5 -> 0, skill=1.0 -> 1
        chosen = []
        for e in range(top_k):
            hit = stable_bool("route:%d:%d:%s" % (layer_id, e, token_key), p=0.5 + 0.5 * boost)
            chosen.append(gold[e] if hit else (gold[e] + e + 1) % self.config.moe.n_experts)
        return chosen

    def generate(self, prompt, max_tokens=16):
        """Deterministic pseudo-completion (no randomness)."""
        toks = prompt.split()[:8]
        return " ".join(toks) + " ->done"

    def train_step(self, delta=0.1):
        """Raise skill toward 1.0 (monotonic)."""
        self.skill = min(1.0, self.skill + delta)
        return self.skill
