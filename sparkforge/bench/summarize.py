"""Run all benchmarks and produce a weighted average."""
from .registry import BENCHMARKS, list_benchmarks
from ..backends import get_backend


def run_all(cfg, backend=None):
    if backend is None:
        backend = get_backend(cfg.model.name, cfg)
    scores = {}
    for name, fn in BENCHMARKS.items():
        try:
            scores[name] = fn(cfg, backend)
        except Exception as e:  # pragma: no cover
            scores[name] = float("nan")
    vals = [v for v in scores.values() if isinstance(v, (int, float)) and v == v]
    avg = sum(vals) / len(vals) if vals else 0.0
    return {"scores": scores, "avg": avg}


def main():
    from ..config import load_config
    cfg = load_config("configs/default.yaml")
    report = run_all(cfg)
    for k, v in report["scores"].items():
        print("  %-16s %.4f" % (k, v))
    print("AVG %.4f" % report["avg"])
