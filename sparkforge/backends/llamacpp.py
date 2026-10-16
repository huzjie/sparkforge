"""llama.cpp / GGUF adapter (declares expected config, falls back offline)."""
from .registry import register_backend


@register_backend("llamacpp")
class LlamaCppBackend:
    def __init__(self, config):
        self.config = config

    def generate(self, prompt, max_tokens=16):
        return " ".join(prompt.split()[:8]) + " ->llamacpp(offline)"
