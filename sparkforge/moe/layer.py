"""One sparse MoE layer: route -> run top-k experts -> weighted sum."""
from ..tensor import Tensor
from .expert import SwiGLUExpert
from .router import Router


class MoELayer:
    """256 experts, activate 8. The sparse forward is the whole point: total
    params ~ n_experts * 3*dim*hidden, active params ~ top_k * 3*dim*hidden.
    """

    def __init__(self, n_experts, top_k, dim, hidden, layer_id=0):
        self.n_experts = n_experts
        self.top_k = top_k
        self.dim = dim
        self.hidden = hidden
        self.layer_id = layer_id
        self.router = Router(n_experts, dim, top_k)
        self.experts = [SwiGLUExpert(e, dim, hidden, layer_id) for e in range(n_experts)]

    def forward(self, x, gate_scale=1.0):
        """x: Tensor (1 x dim) -> Tensor (1 x dim)."""
        top, gates, aux = self.router.route(x)
        acc = Tensor.zeros(1, self.dim)
        norm = 0.0
        for i, eid in enumerate(top):
            w = max(gates[i] * gate_scale, 0.0)
            out = self.experts[eid].forward(x)
            acc = acc + out * w
            norm += w
        if norm > 0:
            acc = acc / norm
        return acc, aux, top

    def active_ratio(self):
        return self.top_k / self.n_experts
