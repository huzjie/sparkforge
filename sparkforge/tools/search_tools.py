"""Search tool (grep) for the coding agent."""
import os


class GrepTool:
    name = "grep"

    def __init__(self, root="."):
        self.root = root

    def __call__(self, pattern, path=None):
        base = os.path.join(self.root, path or "")
        hits = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".venv", "node_modules")]
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                try:
                    with open(full, "r", encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if pattern in line:
                                hits.append("%s:%d: %s" % (os.path.relpath(full, self.root), i, line.strip()))
                except OSError:
                    continue
        return hits
