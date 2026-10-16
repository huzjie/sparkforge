"""Hugging Face Transformers adapter (fallback offline)."""
from .registry import register_backend


@register_backend("transformers")
class TransformersBackend:
    def __init__(self, config):
        self.config = config

    def generate(self, prompt, max_tokens=16):
        return " ".join(prompt.split()[:8]) + " ->hf(offline)"
