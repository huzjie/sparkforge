"""Repo index / navigation / refactor / coding loop."""
import unittest
import tempfile
import os

from sparkforge.agent.repo_index import RepoIndex
from sparkforge.agent.navigator import Navigator
from sparkforge.agent.refactor import RefactorPlanner
from sparkforge.agent.self_correct import SelfCorrector
from sparkforge.agent.coding_loop import CodingAgent
from sparkforge.config import Config


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="sf_repo_")
        with open(os.path.join(self.root, "app.py"), "w", encoding="utf-8") as f:
            f.write("import util\ndef main():\n    return util.helper()\n")
        with open(os.path.join(self.root, "util.py"), "w", encoding="utf-8") as f:
            f.write("def helper():\n    return 42\n")

    def test_index_defs(self):
        idx = RepoIndex(self.root).build()
        self.assertIn("main", idx.by_def)
        self.assertIn("helper", idx.by_def)

    def test_navigator_importers(self):
        idx = RepoIndex(self.root).build()
        nav = Navigator(idx)
        self.assertEqual(len(nav.importers("util")), 1)

    def test_refactor_plan(self):
        idx = RepoIndex(self.root).build()
        planner = RefactorPlanner(idx, 6)
        plan = planner.plan("fix null pointer", ["main"])
        self.assertGreaterEqual(len(plan.edits), 1)

    def test_self_correct(self):
        sc = SelfCorrector(3)
        corr = sc.correct(["a.py"], "test failed", "fix it")
        self.assertIn("previous attempt failed", corr.retry_prompt)
        self.assertEqual(corr.remaining_budget, 2)

    def test_coding_loop(self):
        cfg = Config()
        cfg.model.skill = 1.0
        agent = CodingAgent(cfg, backend=None)
        result = agent.run("refactor main", self.root, ["main"])
        self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
