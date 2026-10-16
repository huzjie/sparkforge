"""Run the agentic coding loop on a synthetic repo."""
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkforge import Config
from sparkforge.agent.coding_loop import CodingAgent

root = tempfile.mkdtemp(prefix="sf_repo_")
for i in range(6):
    with open(os.path.join(root, "mod%d.py" % i), "w", encoding="utf-8") as f:
        f.write("def func_%d():\n    return %d\n" % (i, i))

cfg = Config()
agent = CodingAgent(cfg, backend=None)
result = agent.run("optimize the hot path", root, ["func_0", "func_1"])
print("success:", result["success"])
print("steps:", result["steps"])
print("located:", result.get("located"))
