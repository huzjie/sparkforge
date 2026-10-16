"""Record an agent trajectory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkforge.agent.trajectory import Trajectory

traj = Trajectory("demo task")
traj.record("index repo", "done", tool="index")
traj.record("plan edits", "3 files", tool="plan", ok=True)
print(traj.to_json())
