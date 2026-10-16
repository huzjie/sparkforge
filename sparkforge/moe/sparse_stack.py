"""Stack of MoE layers + a learned router warm-up / load-balance summary."""
from ..tensor import Tensor
from .layer import MoELayer


class SparseStack:
    """A stack of MoE layers. Tracks per-token activated experts and aggregate
    load-balance loss, which feeds the trainer (mock skill -> better routing).
    """

    def __init__(self, n_experts, top_k, dim, hidden, n_layers):
        self.layers = [MoELayer(n_experts, top_k, dim, hidden, i) for i in range(n_layers)]
        self.n_layers = n_layers

    def forward(self, x, gate_scale=1.0):
        aux_total = 0.0
        activated = []
        h = x
        for layer in self.layers:
            h, aux, top = layer.forward(h, gate_scale)
            aux_total += aux
            activated.append(top)
        return h, aux_total, activated

    @property
    def n_experts(self):
        return self.layers[0].n_experts if self.layers else 0

    @property
    def top_k(self):
        return self.layers[0].top_k if self.layers else 0
