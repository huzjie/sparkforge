"""Model wrappers: SparkModel (MoE + multimodal) and generate."""
from .spark import SparkModel
from .generate import generate

__all__ = ["SparkModel", "generate"]
