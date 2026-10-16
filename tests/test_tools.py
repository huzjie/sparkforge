"""Tool registry + file tools."""
import unittest
import tempfile
import os

from sparkforge.tools.registry import ToolRegistry
from sparkforge.tools.build_registry import default_registry


class TestTools(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="sf_tools_")

    def test_registry(self):
        reg = ToolRegistry()
        reg.register("double", lambda x: x * 2, "double a number")
        self.assertEqual(reg.call("double", x=21), 42)
        self.assertIn("double", reg.names())

    def test_default_registry_read_write(self):
        reg = default_registry(self.root)
        reg.call("write_file", path="a.txt", content="hello")
        self.assertEqual(reg.call("read_file", path="a.txt"), "hello")

    def test_grep(self):
        reg = default_registry(self.root)
        reg.call("write_file", path="b.py", content="def foo():\n    return 1\n")
        hits = reg.call("grep", pattern="foo")
        self.assertGreaterEqual(len(hits), 1)


if __name__ == "__main__":
    unittest.main()
