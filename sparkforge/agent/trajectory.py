"""Trace recording for the coding agent (replay + inspection)."""
import json
import time
from dataclasses import dataclass, field


@dataclass
class Step:
    index: int
    prompt: str
    output: str
    tool: str = ""
    ok: bool = True
    note: str = ""


@dataclass
class Trajectory:
    task: str
    steps: list = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def record(self, step, output, tool="", ok=True, note=""):
        self.steps.append(Step(len(self.steps), step, output, tool, ok, note))

    def to_dict(self):
        return {"task": self.task, "started": self.started,
                "steps": [vars(s) for s in self.steps]}

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
