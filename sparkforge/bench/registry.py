"""Benchmark registry."""
BENCHMARKS = {}


def register_benchmark(name):
    def deco(fn):
        BENCHMARKS[name] = fn
        return fn
    return deco


def list_benchmarks():
    return list(BENCHMARKS.keys())
