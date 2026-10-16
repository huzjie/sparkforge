# Benchmarks

| benchmark | what it measures | baseline -> ceiling |
|-----------|------------------|---------------------|
| coding    | task success rate | skill 0.5 -> 1.0 |
| routing   | recall@top-k of gold experts | ~top_k/n -> 1.0 |
| refactor  | plan coverage within budget | > 0 |
| context_recall | addressable blocks after offload | stable |
| bandwidth | normalized tokens/sec | scales with hit rate |

Run `python -m sparkforge.cli bench`.
