"""Deterministic token stream generator for routing/context benchmarks."""
from ..utils.hash import stable_vector


def token_stream(n, dim, seed="stream"):
    """Yield ``n`` deterministic token key/vector pairs."""
    for i in range(n):
        yield "tok-%d" % i, stable_vector("%s:%d" % (seed, i), dim)
