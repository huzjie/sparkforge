"""Generate a synthetic repo and run the agent on it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkforge.data.synth_repo import synth_repo
from sparkforge.config import Config
from sparkforge.agent.coding_loop import CodingAgent

root = synth_repo(n_files=12, n_defs_per_file=3)
agent = CodingAgent(Config(), backend=None)
result = agent.run("optimize hot path", root, ["func_2_0"])
print("success:", result["success"], "steps:", result["steps"])
