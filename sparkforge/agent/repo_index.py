"""A lightweight AST/name index of a codebase.

Builds a file graph (imports + definitions) so the agent can find "where is X
defined / who imports X" without reading every file -- the retrieval step that
makes agentic coding on a 1M-token repo tractable.
"""
import ast
import os
from dataclasses import dataclass, field


@dataclass
class FileNode:
    path: str
    imports: list = field(default_factory=list)
    defs: list = field(default_factory=list)
    n_lines: int = 0


class RepoIndex:
    def __init__(self, root, limit=256):
        self.root = root
        self.limit = limit
        self.files = {}   # path -> FileNode
        self.by_def = {}  # name -> [paths]

    def build(self):
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".venv", "node_modules")]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, self.root).replace(os.sep, "/")
                self._index_file(full, rel)
                if len(self.files) >= self.limit:
                    return self
        return self

    def _index_file(self, full, rel):
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                src = f.read()
        except OSError:
            return
        node = FileNode(path=rel, n_lines=src.count("\n") + 1)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            node.defs = ["<unparsable>"]
        else:
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    for a in n.names:
                        node.imports.append(a.name)
                elif isinstance(n, ast.ImportFrom):
                    node.imports.append(n.module or "")
                elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    node.defs.append(n.name)
                    self.by_def.setdefault(n.name, []).append(rel)
        self.files[rel] = node

    def find_def(self, name):
        return self.by_def.get(name, [])

    def importers_of(self, module):
        return [rel for rel, node in self.files.items()
                if module in node.imports or any(imp.startswith(module + ".") for imp in node.imports)]
