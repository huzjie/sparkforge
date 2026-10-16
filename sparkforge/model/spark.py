"""SparkModel: the sparse MoE + multimodal + context model glue."""
from ..config import Config
from ..moe.sparse_stack import SparseStack
from ..multimodal.fusion import GatedFusion
from ..tensor import Tensor, layernorm


class SparkModel:
    """Wraps the MoE stack + multimodal fusion. The token embedding and the MoE
    output head are deterministic here; the value is the *architecture shape*
    (sparse forward + native multimodal + paged context), not learned weights.
    """

    def __init__(self, config: Config):
        self.config = config
        m = config.moe
        self.stack = SparseStack(m.n_experts, m.top_k, m.dim, m.hidden, m.n_layers)
        self.fusion = GatedFusion(m.dim)
        self.dim = m.dim

    def embed(self, text):
        """Deterministic bag-of-token embedding (1 x dim)."""
        from ..utils.hash import stable_float
        toks = text.split()
        if not toks:
            toks = ["<empty>"]
        row = [sum(stable_float("emb:%d:%s" % (j, t), -1, 1) for t in toks) / len(toks)
               for j in range(self.dim)]
        return Tensor([row])

    def forward(self, text, media_emb=None):
        """text -> token embedding -> MoE stack -> (optional) media fusion."""
        x = self.embed(text)
        if media_emb is not None:
            x = self.fusion.fuse(x, media_emb)
        out, aux, activated = self.stack.forward(x)
        return layernorm(out), aux, activated

    def logits_for(self, output):
        """Project the final hidden state to a pseudo-vocab score vector."""
        from ..utils.hash import stable_float
        row = output.data[0]
        return [stable_float("head:%d:%s" % (i, "|".join("%.2f" % v for v in row)), -2, 2)
                for i in range(64)]
