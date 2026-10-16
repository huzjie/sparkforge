"""Sparse mixture-of-experts (256 experts / top-8 routing)."""
from .expert import SwiGLUExpert
from .router import Router
from .layer import MoELayer

__all__ = ["SwiGLUExpert", "Router", "MoELayer"]
