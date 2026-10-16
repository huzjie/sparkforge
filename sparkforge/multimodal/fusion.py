"""Gated fusion: combine text + media embeddings into one token stream."""
from ..tensor import Tensor, layernorm
from ..utils.hash import stable_float


class GatedFusion:
    """Projects each modality embedding to a shared dim, then a learned gate
    (deterministic here) blends them. Native multimodality = one shared space,
    not late concatenation.
    """

    def __init__(self, dim):
        self.dim = dim
        self.gate_bias = stable_float("fusion:bias", -0.5, 0.5)

    def fuse(self, text_emb, media_emb):
        """text_emb, media_emb: Tensor (1 x dim). Returns blended Tensor."""
        g = self._gate(text_emb)
        blended = text_emb * g + media_emb * (1.0 - g)
        return layernorm(blended)

    def _gate(self, text_emb):
        row = text_emb.data[0]
        s = sum(row) / len(row)
        # sigmoid
        import math
        return 1.0 / (1.0 + math.exp(-(s + self.gate_bias)))
