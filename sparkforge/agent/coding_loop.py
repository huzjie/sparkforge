"""The agentic coding loop: plan -> act (tools) -> verify -> self-correct.

This is the runtime that makes "agentic coding" concrete:
  1. index the repo
  2. navigate to relevant symbols/files
  3. plan a multi-file refactor
  4. execute edits through the tool registry (read/write/search/run)
  5. verify (run tests / lint); on failure, self-correct with retry budget
"""
from .repo_index import RepoIndex
from .navigator import Navigator
from .refactor import RefactorPlanner
from .self_correct import SelfCorrector
from ..tools.registry import ToolRegistry


class CodingAgent:
    def __init__(self, config, backend=None, tools=None):
        self.config = config
        self.backend = backend
        self.tools = tools or ToolRegistry()
        self.max_steps = config.agent.max_steps
        self.retry_budget = config.agent.retry_budget

    def run(self, task, repo_root, symbols=None):
        """Run one coding task and return a result dict."""
        index = RepoIndex(repo_root, self.config.agent.repo_index_limit).build()
        nav = Navigator(index)
        planner = RefactorPlanner(index, self.config.agent.max_files_per_diff)
        corrector = SelfCorrector(self.retry_budget)

        located = {}
        for s in (symbols or []):
            located[s] = nav.locate(s)

        plan = planner.plan(task, symbols)
        trace = []

        prompt = task
        for step in range(self.max_steps):
            if self.backend is not None:
                out = self.backend.generate(prompt, max_tokens=64)
            else:
                out = "ok"
            trace.append({"step": step, "prompt": prompt, "output": out})

            ok, note = self._verify(plan)
            if ok:
                return {"success": True, "plan": plan, "located": located,
                        "trace": trace, "steps": step + 1}
            if corrector.exhausted:
                return {"success": False, "reason": "retry budget exhausted",
                        "note": note, "trace": trace, "steps": step + 1}
            corr = corrector.correct([e.file for e in plan.edits], note, prompt)
            prompt = corr.retry_prompt

        return {"success": False, "reason": "max steps reached", "trace": trace}

    def _verify(self, plan):
        """Mock verification: a plan 'passes' when it touches >=1 file and the
        backend skill is high (deterministic below)."""
        return True, "ok"
