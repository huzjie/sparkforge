"""Evaluation metrics (accuracy, precision, recall, F1, mse)."""
import math


def accuracy(preds, targets):
    if not preds:
        return 0.0
    return sum(1 for p, t in zip(preds, targets) if p == t) / len(preds)


def precision(preds, targets, positive=True):
    tp = sum(1 for p, t in zip(preds, targets) if p == positive and t == positive)
    fp = sum(1 for p, t in zip(preds, targets) if p == positive and t != positive)
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall(preds, targets, positive=True):
    tp = sum(1 for p, t in zip(preds, targets) if p == positive and t == positive)
    fn = sum(1 for p, t in zip(preds, targets) if p != positive and t == positive)
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1(preds, targets, positive=True):
    p = precision(preds, targets, positive)
    r = recall(preds, targets, positive)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def mse(a, b):
    if not a:
        return 0.0
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def rmse(a, b):
    return math.sqrt(mse(a, b))
