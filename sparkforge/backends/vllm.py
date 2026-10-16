"""vLLM backend adapter (declares expected config, falls back offline)."""
from .registry import register_backend


@register_backend("vllm")
class VllmBackend:
    def __init__(self, config):
        self.config = config

    def generate(self, prompt, max_tokens=16):
        return " ".join(prompt.split()[:8]) + " ->vllm(offline)"
