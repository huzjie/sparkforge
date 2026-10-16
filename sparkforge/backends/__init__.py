"""Backend registry: mock (trainable) + cpu + real-model adapters."""
from .registry import register_backend, get_backend, list_backends
from . import mock, cpu, openai, vllm, transformers, llamacpp

__all__ = ["register_backend", "get_backend", "list_backends"]
