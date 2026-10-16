# Refactor planner prompt

Task: {task}
Symbols to focus on: {symbols}
Repo index summary (defs -> files): {index_summary}

Produce an ordered plan of file edits. For each edit, specify:
- file (relative path)
- action (modify | add | delete)
- reason (why this edit advances the task)

Limit to {max_files} files. Flag risk as low/medium/high.
