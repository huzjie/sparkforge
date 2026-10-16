# Configuration reference

| key | default | meaning |
|-----|---------|---------|
| moe.n_experts | 256 | experts per layer |
| moe.top_k | 8 | active experts per token |
| moe.total_params_b | 816.0 | declared total params (billions) |
| moe.active_params_b | 42.0 | declared active params (billions) |
| memory.bandwidth_gbps | 800.0 | unified-memory bandwidth |
| memory.prefetch_depth | 8 | lookahead depth |
| context.block_size | 128 | KV block size (tokens) |
| context.gpu_blocks | 16 | resident block budget |
| context.offload | true | enable LRU offload |
| agent.retry_budget | 3 | self-correction retries |
| model.skill | 0.5 | mock router skill |
