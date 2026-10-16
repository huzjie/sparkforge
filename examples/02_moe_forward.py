"""Run a sparse MoE forward and inspect routing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkforge import Config
from sparkforge.model.spark import SparkModel

model = SparkModel(Config())
out, aux, activated = model.forward("refactor the billing module")
print("output dim:", len(out.data[0]))
print("aux loss:", round(aux, 4))
print("layer 0 activated experts:", activated[0])
