# Public API

```python
from sparkforge import Config, SparkEngine, get_backend

cfg = Config()
eng = SparkEngine(cfg)
eng.complete("def add(a, b):")          # generate
eng.code("refactor", ".", ["main"])     # agentic coding
eng.memory_report()                     # tiers

backend = get_backend("mock", cfg)      # trainable
backend.train_step(0.1)                 # raise skill
```

Backends: `mock`, `cpu`, `openai`, `vllm`, `transformers`, `llamacpp`.
