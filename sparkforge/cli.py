"""Command-line interface: bench / code / serve / doctor / memory."""
import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sparkforge", description="Unified-memory MoE coding-agent engine")
    ap.add_argument("--config", default="configs/default.yaml", help="path to config YAML")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("doctor", help="self-check components")
    sub.add_parser("bench", help="run benchmarks")
    sub.add_parser("train", help="train the mock router")
    sub.add_parser("serve", help="start HTTP server")
    sub.add_parser("memory", help="print memory tiers")

    code = sub.add_parser("code", help="run a coding task")
    code.add_argument("--prompt", required=True)
    code.add_argument("--repo", default=".")
    code.add_argument("--symbols", nargs="*", default=[])

    args = ap.parse_args(argv)

    from .config import load_config
    cfg = load_config(args.config)

    if args.cmd == "doctor":
        from .doctor import main as d
        return d()
    if args.cmd == "bench":
        from .bench.summarize import run_all
        from .backends import get_backend
        backend = get_backend(cfg.model.name, cfg)
        report = run_all(cfg, backend)
        for k, v in report["scores"].items():
            print("  %-16s %.4f" % (k, v))
        print("AVG %.4f" % report["avg"])
        return 0
    if args.cmd == "train":
        from .train import train
        backend, hist = train(cfg, steps=5)
        for k, v in hist:
            print("  %-6s %.4f" % (k, v))
        return 0
    if args.cmd == "serve":
        from .serving import run
        run(cfg)
        return 0
    if args.cmd == "memory":
        from .engine import SparkEngine
        eng = SparkEngine(cfg)
        print(eng.memory_report())
        return 0
    if args.cmd == "code":
        from .engine import SparkEngine
        eng = SparkEngine(cfg)
        result = eng.code(args.prompt, args.repo, args.symbols)
        print("success=%s steps=%s" % (result.get("success"), result.get("steps")))
        if not result.get("success"):
            print("reason=%s" % result.get("reason"))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
