"""Run all benchmarks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkforge.config import load_config
from sparkforge.bench.summarize import run_all
from sparkforge.backends import get_backend

cfg = load_config("configs/default.yaml")
backend = get_backend(cfg.model.name, cfg)
report = run_all(cfg, backend)
for k, v in report["scores"].items():
    print("  %-16s %.4f" % (k, v))
print("AVG %.4f" % report["avg"])
