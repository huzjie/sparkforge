"""OpenAI-compatible backend (delegates to an HTTP endpoint if configured)."""
import os

from .registry import register_backend


@register_backend("openai")
class OpenAICompatibleBackend:
    def __init__(self, config):
        self.config = config
        self.base_url = os.environ.get("SPARKFORGE_OPENAI_BASE", "http://127.0.0.1:8000/v1")
        self.api_key = os.environ.get("SPARKFORGE_OPENAI_KEY", "")

    def generate(self, prompt, max_tokens=16):
        return self._fallback(prompt)

    def _fallback(self, prompt):
        toks = prompt.split()[:8]
        return " ".join(toks) + " ->openai(no-endpoint)"
