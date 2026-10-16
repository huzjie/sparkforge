"""A single SwiGLU expert (small MLP)."""
from ..tensor import Tensor, matmul
from ..utils.hash import stable_vector


class SwiGLUExpert:
    """One feed-forward expert: SwiGLU(x) = (W2 @ (swish(W1 x) * (W3 x))).

    Weights are derived deterministically from the expert id so the 256-expert
    layer stays lightweight and reproducible without storing real checkpoints.
    """

    def __init__(self, expert_id, dim, hidden, layer_id=0):
        self.expert_id = expert_id
        self.dim = dim
        self.hidden = hidden
        seed_key = "expert:%d:%d" % (layer_id, expert_id)
        self.w1 = Tensor([stable_vector("%s:w1:%d" % (seed_key, j), dim) for j in range(hidden)])
        self.w3 = Tensor([stable_vector("%s:w3:%d" % (seed_key, j), dim) for j in range(hidden)])
        self.w2 = Tensor([stable_vector("%s:w2:%d" % (seed_key, j), hidden) for j in range(dim)])

    @staticmethod
    def _swish(v):
        # numerically stable swish = v * sigmoid(v)
        import math
        if v >= 0:
            return v / (1.0 + math.exp(-v))
        e = math.exp(v)
        return v * e / (1.0 + e)

    def forward(self, x):
        """x: Tensor (1 x dim) -> Tensor (1 x dim)."""
        row = x.data[0]
        h1 = [sum(row[k] * self.w1.data[j][k] for k in range(self.dim)) for j in range(self.hidden)]
        h3 = [sum(row[k] * self.w3.data[j][k] for k in range(self.dim)) for j in range(self.hidden)]
        gate = [self._swish(h1[j]) * h3[j] for j in range(self.hidden)]
        out = [sum(gate[j] * self.w2.data[i][j] for j in range(self.hidden)) for i in range(self.dim)]
        return Tensor([out])
