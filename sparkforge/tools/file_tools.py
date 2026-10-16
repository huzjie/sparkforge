"""File read/write tools for the coding agent."""
import os


class ReadFileTool:
    name = "read_file"

    def __init__(self, root="."):
        self.root = root

    def __call__(self, path):
        full = os.path.join(self.root, path)
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


class WriteFileTool:
    name = "write_file"

    def __init__(self, root="."):
        self.root = root

    def __call__(self, path, content):
        full = os.path.join(self.root, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return {"written": path, "bytes": len(content)}
