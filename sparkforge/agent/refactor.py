"""Refactor planning: turn a task into an ordered plan of file edits."""
from dataclasses import dataclass, field


@dataclass
class Edit:
    file: str
    action: str  # "modify" | "add" | "delete"
    reason: str
    snippet: str = ""


@dataclass
class RefactorPlan:
    task: str
    edits: list = field(default_factory=list)
    risk: str = "low"

    def to_summary(self):
        return [("edit %s %s" % (e.action, e.file)) for e in self.edits]


class RefactorPlanner:
    """Deterministic planner: for the mock it derives a sensible multi-file plan
    from the task and the index, so the coding loop has something real to execute.
    """

    def __init__(self, index, max_files_per_diff=6):
        self.index = index
        self.max_files_per_diff = max_files_per_diff

    def plan(self, task, symbols=None):
        symbols = symbols or []
        files = self._candidate_files(symbols)
        edits = []
        for i, f in enumerate(files[: self.max_files_per_diff]):
            edits.append(Edit(file=f, action="modify",
                              reason="touchpoint for: %s" % task,
                              snippet="# sparkforge: %s" % task))
        if not edits:
            edits.append(Edit(file="src/main.py", action="add",
                              reason="new entrypoint for: %s" % task))
        risk = "high" if len(edits) >= self.max_files_per_diff else "low"
        return RefactorPlan(task=task, edits=edits, risk=risk)

    def _candidate_files(self, symbols):
        files = []
        for s in symbols:
            files.extend(self.index.find_def(s))
        if not files:
            files = list(self.index.files.keys())[: self.max_files_per_diff]
        return files
