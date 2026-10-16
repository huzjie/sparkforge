# Recipe — agentic bug-fix runbook

1. `sparkforge code --prompt "fix the null-pointer in utils.py" --repo . --symbols utils`
2. The agent indexes the repo, locates `utils`, plans an edit, writes it, then
   runs the test suite.
3. If a test fails, it rolls back and retries with the failure note (retry budget
   from config `agent.retry_budget`).
4. Inspect the trace to see the prompt evolution.

Tips: raise `agent.max_steps` for long debugging sessions; set `--symbols` to the
functions you already know are involved to skip index misses.
