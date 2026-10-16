"""Train the mock router and show the benchmark drift."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkforge.config import load_config
from sparkforge.train import train
from sparkforge.bench.summarize import run_all
from sparkforge.backends import get_backend

cfg = load_config("configs/default.yaml")
backend, hist = train(cfg, steps=5, delta=0.1)
print("skill:", hist[0][1], "->", hist[-1][1])
report = run_all(cfg, backend)
print("routing recall after train: %.4f" % report["scores"].get("routing", 0.0))
