"""Assemble a default tool registry wired to a repo root."""
from .registry import ToolRegistry
from .file_tools import ReadFileTool, WriteFileTool
from .search_tools import GrepTool
from .exec_tools import RunTool


def default_registry(root="."):
    reg = ToolRegistry()
    reg.register("read_file", ReadFileTool(root), "Read a file by relative path.")
    reg.register("write_file", WriteFileTool(root), "Write content to a file.")
    reg.register("grep", GrepTool(root), "Search for a pattern in the repo.")
    reg.register("run", RunTool(root), "Run a shell command.")
    return reg
