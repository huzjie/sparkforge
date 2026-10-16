"""Navigation over a repo index: locate definitions, importers, related files."""
import os

from .repo_index import RepoIndex


class Navigator:
    def __init__(self, index: RepoIndex):
        self.index = index

    def locate(self, symbol):
        """Return ranked candidate files defining ``symbol``."""
        return self.index.find_def(symbol)

    def importers(self, module):
        return self.index.importers_of(module)

    def read(self, path):
        full = os.path.join(self.index.root, path)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return None
