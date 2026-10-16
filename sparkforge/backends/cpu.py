"""CPU reference backend: runs the actual SparseStack (tiny config)."""
from .registry import register_backend
from ..model.spark import SparkModel


@register_backend("cpu")
class CpuBackend:
    def __init__(self, config):
        self.config = config
        self.model = SparkModel(config)

    def generate(self, prompt, max_tokens=16):
        from ..model.generate import generate
        return generate(self.model, prompt, max_tokens)

    def forward(self, text, media_emb=None):
        return self.model.forward(text, media_emb)
