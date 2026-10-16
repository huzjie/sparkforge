"""Coding benchmark: task success rate scales with routing skill."""
from .registry import register_benchmark
from ..oracle import coding_correct


@register_benchmark("coding")
def run(cfg, backend=None, n_tasks=40):
    skill = backend.skill if backend is not None else cfg.model.skill
    hits = sum(1 for i in range(n_tasks) if coding_correct("task-%d" % i, skill))
    return hits / n_tasks
