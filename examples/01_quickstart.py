"""Quickstart: run inference and a coding task."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkforge import Config, SparkEngine

cfg = Config()
eng = SparkEngine(cfg)
print("complete:", eng.complete("def add(a, b):"))
print("memory:", eng.memory_report())
