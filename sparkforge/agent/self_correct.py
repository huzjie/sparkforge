"""Self-correction: detect a bad outcome and roll back / retry with budget.

Long debugging sessions succeed by *recovering* from mistakes, not by never
making them. The corrector maintains a retry budget; on failure it rolls back the
last edit and retries with the failure note appended to the prompt (a concrete
form of "correct itself over long tasks").
"""
from dataclasses import dataclass, field


@dataclass
class Correction:
    rollback_edits: list
    retry_prompt: str
    remaining_budget: int


class SelfCorrector:
    def __init__(self, retry_budget=3):
        self.retry_budget = retry_budget
        self.history = []

    def correct(self, failed_edits, failure_note, prompt):
        """Return a Correction with a fresh prompt that carries the failure note."""
        self.history.append({"failed_edits": failed_edits, "note": failure_note})
        remaining = self.retry_budget - len(self.history)
        retry = ("%s\n\n[previous attempt failed: %s]\nFix the above and retry."
                 % (prompt, failure_note))
        return Correction(rollback_edits=failed_edits, retry_prompt=retry,
                          remaining_budget=remaining)

    @property
    def exhausted(self):
        return len(self.history) >= self.retry_budget
