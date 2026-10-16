# Architecture

sparkforge is a zero-dependency reference implementation of the mechanisms behind
Meta's **Muse Spark Open**: an 816B native-multimodal sparse-MoE model built for
agentic coding, runnable on unified-memory machines.

## Layers

```
+----------------------+    +----------------------+    +----------------------+
|  multimodal encoders |    |  sparse MoE stack    |    |  paged KV cache      |
|  vision/audio/video  | -> |  256 experts / top-8 | -> |  1M-token offload    |
+----------------------+    +----------------------+    +----------------------+
            |                         |                         |
            v                         v                         v
+----------------------------------------------------------------------+
|  unified-memory scheduler (tiers + bandwidth-aware prefetch)          |
+----------------------------------------------------------------------+
            |
            v
+----------------------------------------------------------------------+
|  agentic coding loop: index -> navigate -> plan -> tools -> correct   |
+----------------------------------------------------------------------+
```

## Key mechanisms

1. **Sparse forward** — `moe/`: a gated `Router` scores 256 experts and activates
   top-8; `MoELayer` runs only those, so compute scales with the *active* 42B,
   not the *total* 816B.
2. **Unified memory** — `memory/`: `MemoryTiers` (GPU/unified/SSD) + a
   bandwidth-aware `UnifiedMemoryScheduler` + a `PrefetchPipeline` that streams
   next-layer experts during compute.
3. **Native multimodal** — `multimodal/`: vision patches, audio frames, sampled
   video frames all project into one token space and interleave into context.
4. **1M context** — `context/`: paged `KVCache` + `BlockManager` offload LRU
   blocks and swap them back on demand.
5. **Agentic coding** — `agent/`: `RepoIndex` (defs + importers), `Navigator`,
   `RefactorPlanner`, `SelfCorrector`, `CodingAgent`.

See `docs/MECHANISMS.md` for the "why" behind each.
