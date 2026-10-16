"""Gated router: scores all experts, picks top-k, emits load-balance aux loss."""
from ..tensor import Tensor
from ..utils.hash import stable_float


class Router:
    """Scores every expert with a linear gate ``logit = <w_e, x>`` (deterministic
    here) and selects the top-k. Also reports a load-balance auxiliary loss so a
    trainer can discourage expert collapse (all tokens hitting one expert).
    """

    def __init__(self, n_experts, dim, top_k):
        self.n_experts = n_experts
        self.dim = dim
        self.top_k = top_k

    def logits(self, x):
        row = x.data[0]
        return [stable_float("gate:%d:%s" % (e, "|".join("%.2f" % v for v in row)), -2.0, 2.0)
                for e in range(self.n_experts)]

    def route(self, x):
        """Return (top_k expert ids, top_k gates, load-balance aux loss)."""
        logits = self.logits(x)
        order = sorted(range(self.n_experts), key=lambda i: (-logits[i], i))
        top = order[: self.top_k]
        gates = [logits[i] for i in top]
        # load balance: penalize deviation of each expert's selection fraction
        # from uniform (1/n). selection fraction ~ 1 if in top_k else 0.
        n = self.n_experts
        frac = [1.0 if e in top else 0.0 for e in range(n)]
        uniform = 1.0 / n
        aux = sum(abs(frac[e] - uniform) for e in range(n))
        return top, gates, aux
