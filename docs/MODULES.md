# Module map

- `sparkforge/moe/` — Router, SwiGLUExpert, MoELayer, SparseStack
- `sparkforge/memory/` — MemoryTiers, UnifiedMemoryScheduler, PrefetchPipeline, bandwidth
- `sparkforge/multimodal/` — VisionEncoder, AudioEncoder, VideoEncoder, InterleavedSequence, GatedFusion
- `sparkforge/context/` — KVCache, BlockManager
- `sparkforge/agent/` — RepoIndex, Navigator, RefactorPlanner, SelfCorrector, CodingAgent, Trajectory
- `sparkforge/tools/` — ToolRegistry, file/search/exec tools, build_registry
- `sparkforge/model/` — SparkModel, generate
- `sparkforge/backends/` — mock/cpu/openai/vllm/transformers/llamacpp
- `sparkforge/eval/` — routing_eval, coding_eval, capacity_eval
- `sparkforge/data/` — synth_repo, token_stream
