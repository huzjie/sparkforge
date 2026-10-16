"""Mock trainer: raise routing skill monotonically; show benchmark drift.

The signal is intentionally monotonic (skill always climbs), which mirrors the
"active params help the most when the router is good" story and keeps the demo
clean without needing a real optimizer.
"""
from .config import load_config
from .backends import get_backend
from .bench import run_all


def train(cfg, steps=5, delta=0.1):
    backend = get_backend(cfg.model.name, cfg)
    hist = [("init", backend.skill)]
    for step in range(steps):
        backend.train_step(delta)
        hist.append((step + 1, backend.skill))
    return backend, hist


def main():
    cfg = load_config("configs/default.yaml")
    backend, hist = train(cfg, steps=5)
    print("[train] skill trajectory:")
    for k, v in hist:
        print("  %-6s %.4f" % (k, v))
    report = run_all(cfg)
    print("[train] final bench AVG=%.4f" % report["avg"])


if __name__ == "__main__":
    main()
