"""Single source of truth for gold routing / coding correctness.

Both the mock backend and the benchmarks derive ground truth from this module so
that skill=1.0 is provably optimal -- the shared-oracle pattern that keeps the
mock trainable and the benchmark numbers honest.
"""
from .utils.hash import stable_ints


def gold_experts(token_key, layer_id, top_k, n_experts=None):
    """Deterministic gold top-k experts for a (token, layer) pair (distinct)."""
    import hashlib
    import random
    n = n_experts or 256
    d = hashlib.md5(("gold:%d:%s" % (layer_id, token_key)).encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(d[:8], "big"))
    return rng.sample(range(n), min(top_k, n))


def coding_correct(task_id, skill):
    """Deterministic coding-success flag for a task at a given skill."""
    from .utils.hash import stable_float
    return stable_float("coding:%s" % task_id) < skill
