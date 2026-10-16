"""Generate a synthetic multi-file repo for agent/refactor benchmarks."""
import os
import tempfile


def synth_repo(n_files=12, n_defs_per_file=3):
    """Create a temp repo with ``n_files`` modules, each defining functions.

    Deterministic content so benchmarks are reproducible. Returns the root path.
    """
    root = tempfile.mkdtemp(prefix="sparkforge_repo_")
    for i in range(n_files):
        mod = "mod_%d" % i
        lines = []
        # each file imports a couple of prior modules (builds a dependency graph)
        imports = []
        if i >= 2:
            imports = ["mod_%d" % (i - 1), "mod_%d" % (i - 2)]
        for imp in imports:
            lines.append("import %s" % imp)
        for d in range(n_defs_per_file):
            lines.append("")
            lines.append("def func_%d_%d(x):" % (i, d))
            lines.append("    return x + %d" % (i * n_defs_per_file + d))
        with open(os.path.join(root, mod + ".py"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return root
