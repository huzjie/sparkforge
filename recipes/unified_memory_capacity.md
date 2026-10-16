# Recipe — capacity planning for unified memory

Use `python -m sparkforge.memory_report` to see the declared 816B scale vs
fp16 / nvfp4 footprints. The key number: 816B at 4-bit ~ 381 GB, which fits a
512 GB unified-memory pool. Stream only misses via the scheduler; tune
`memory.prefetch_depth` to overlap transfer with compute.
