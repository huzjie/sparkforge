"""Deterministic greedy generation loop."""
from ..tensor import Tensor


def generate(model, text, max_tokens=16, media_emb=None):
    """Greedy token generation: repeatedly take argmax logit, append token text.

    Deterministic so it is testable; returns the generated suffix string.
    """
    out, _, _ = model.forward(text, media_emb)
    generated = []
    for _ in range(max_tokens):
        logits = model.logits_for(out)
        best = max(range(len(logits)), key=lambda i: (logits[i], -i))
        generated.append("t%d" % best)
        # feed the last token back (degenerate but stable)
        out, _, _ = model.forward(text + " " + " ".join(generated), media_emb)
    return " ".join(generated)
