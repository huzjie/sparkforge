"""sparkforge - unified-memory MoE coding-agent inference engine.

A zero-dependency reference implementation of the core mechanisms behind
Meta's **Muse Spark Open** (816B-parameter native-multimodal mixture-of-experts
model built for agentic coding, 42B active params, 1M-token context, tuned to
run locally on a pair of Apple M5 Ultra Mac Studios through the MLX unified
memory architecture):

* **Sparse MoE (256 experts / top-8 routing)**: a gated router scores every
  expert and activates only 8 per token, so an 816B model spends only ~42B
  parameters of compute per token.

* **Unified-memory scheduling**: every expert lives in one addressable pool
  (CPU + GPU share it, no copies). A bandwidth-aware scheduler prefetches the
  top-8 experts before each token, hiding CPU/SSD latency behind compute -- the
  trick that lets a TB-scale model run on two 512 GB Mac Studios.

* **Native multimodal encoding**: vision patches (screenshots / UI designs),
  audio frames (voice notes) and sampled video frames (screen recordings) are
  projected into the same token space and interleaved into the 1M context.

* **Agentic coding loop**: repo index -> navigation -> refactor plan -> tool
  calls (read/write/search/run) -> self-correction with a retry budget.

Public API:
    from sparkforge import SparkEngine, Config, get_backend
"""

__version__ = "0.1.0"

from .config import Config, load_config
from .engine import SparkEngine
from .backends import get_backend, list_backends

__all__ = [
    "Config",
    "load_config",
    "SparkEngine",
    "get_backend",
    "list_backends",
    "__version__",
]
