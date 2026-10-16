# Runbook

## Install / run (zero dependency)

```bash
python -m compileall -q sparkforge        # compile check
python -m unittest discover -s tests -t . # tests
python -m sparkforge.doctor               # self-check + bench
```

## CLI

```bash
python -m sparkforge.cli bench --config configs/default.yaml
python -m sparkforge.cli train --config configs/default.yaml
python -m sparkforge.cli code --prompt "fix the null-pointer" --repo . --symbols util
python -m sparkforge.cli serve --config configs/default.yaml
python -m sparkforge.memory_report
```

## Config

`configs/default.yaml` is the small fast proxy; declared scale (816B / 42B) lives
in `moe.total_params_b` / `moe.active_params_b` so the *architecture* is honest
without instantiating real tensors. `configs/macstudio.yaml` models the dual
M5 Ultra unified-memory profile.
