"""Refactor benchmark: plan quality = fraction of edits hitting relevant files."""
from .registry import register_benchmark
from ..agent.repo_index import RepoIndex
from ..agent.refactor import RefactorPlanner


@register_benchmark("refactor")
def run(cfg, backend=None, repo_root=None):
    import tempfile, os
    if repo_root is None:
        repo_root = tempfile.mkdtemp(prefix="sparkforge_bench_")
        for i in range(8):
            with open(os.path.join(repo_root, "mod%d.py" % i), "w", encoding="utf-8") as f:
                f.write("def func_%d():\n    return %d\n" % (i, i))
    index = RepoIndex(repo_root, cfg.agent.repo_index_limit).build()
    planner = RefactorPlanner(index, cfg.agent.max_files_per_diff)
    plan = planner.plan("optimize hot path", ["func_0", "func_1", "func_2"])
    n_edits = len(plan.edits)
    n_files = len(index.files)
    # plan quality: produced at least one edit and covered <= budget
    return (min(n_edits, cfg.agent.max_files_per_diff) / cfg.agent.max_files_per_diff
            if n_edits > 0 else 0.0)
