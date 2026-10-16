# Mechanisms — why each piece exists

## Why sparse MoE?

An 816B dense model costs 816B parameters of compute *per token*. A MoE with 256
experts and top-8 routing activates only ~42B per token — ~19x less compute.
`active_ratio() = top_k / n_experts` is the headline number; the router quality
determines whether the saved compute is *worth it* (see the routing benchmark).

## Why unified memory?

Host<->device copies dominate latency in a normal setup. Unified memory removes
the copy: the cost of "loading" an expert is just streaming its bytes at the
memory bandwidth (M5 Ultra ~1.2 TB/s). `UnifiedMemoryScheduler` models this as
`bytes / bandwidth` and shows that prefetch hides it behind compute.

## Why paged KV cache + offload?

A 1M-token context can't fit KV in GPU memory. Paging into blocks + offloading
LRU blocks to unified memory/SSD lets you keep the window addressable; swapping
cold blocks back on demand keeps recall high. `BlockManager` implements the
eviction + swap-back path.

## Why native multimodal?

Separate vision/audio/video towers + late fusion can't share attention across
modalities. Projecting every modality into *one* token space and interleaving
media tokens into the context lets a single decoder attend over text + screenshot
+ voice note together — which is what "read a screenshot while coding" requires.

## Why an agent loop?

Long coding tasks succeed by *recovering* from mistakes. `SelfCorrector` rolls
back the last edit and retries with the failure note appended, capped by a retry
budget. That is the concrete form of "correct itself over long tasks".
