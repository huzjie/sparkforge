"""Demonstrate metrics helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkforge.utils.metrics import accuracy, precision, recall, f1, mse

preds = [1, 1, 0, 1, 0]
targets = [1, 0, 0, 1, 1]
print("accuracy:", accuracy(preds, targets))
print("precision:", precision(preds, targets))
print("recall:", recall(preds, targets))
print("f1:", f1(preds, targets))
print("mse:", mse([1.0, 2.0], [1.1, 2.2]))
