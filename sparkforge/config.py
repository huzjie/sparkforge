"""Configuration dataclass + zero-dependency YAML loader.

When PyYAML is absent (the managed Python has no third-party deps), we fall back
to :mod:`sparkforge.utils.yamlish`.
"""
from dataclasses import dataclass, field
from pathlib import Path

from .utils.yamlish import parse

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class MoEConfig:
    n_experts: int = 256
    top_k: int = 8
    dim: int = 32
    hidden: int = 64
    n_layers: int = 6
    load_balance_coef: float = 0.01
    # Declared scale (not instantiated): Muse Spark Open.
    total_params_b: float = 816.0
    active_params_b: float = 42.0


@dataclass
class MemoryConfig:
    bandwidth_gbps: float = 800.0
    gpu_pool_gb: float = 64.0
    unified_pool_gb: float = 512.0
    ssd_pool_gb: float = 1024.0
    prefetch_depth: int = 8


@dataclass
class MultimodalConfig:
    patch_size: int = 8
    img_size: int = 32
    n_audio_frames: int = 8
    n_video_frames: int = 4
    dim: int = 32


@dataclass
class ContextConfig:
    max_tokens: int = 1048576
    block_size: int = 128
    gpu_blocks: int = 16
    offload: bool = True


@dataclass
class AgentConfig:
    max_steps: int = 12
    retry_budget: int = 3
    repo_index_limit: int = 256
    max_files_per_diff: int = 6


@dataclass
class ModelConfig:
    name: str = "mock"
    dim: int = 32
    skill: float = 0.5


@dataclass
class ServingConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class Config:
    moe: MoEConfig = field(default_factory=MoEConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    seed: int = 42

    @classmethod
    def from_dict(cls, d):
        def i(v, default):
            return int(v) if v is not None else default

        def f(v, default):
            return float(v) if v is not None else default

        def s(v, default):
            return str(v) if v is not None else default

        def b(v, default):
            return bool(v) if v is not None else default

        moe = d.get("moe", {}) or {}
        memory = d.get("memory", {}) or {}
        mm = d.get("multimodal", {}) or {}
        ctx = d.get("context", {}) or {}
        agent = d.get("agent", {}) or {}
        model = d.get("model", {}) or {}
        serving = d.get("serving", {}) or {}
        return cls(
            moe=MoEConfig(
                n_experts=i(moe.get("n_experts"), 256),
                top_k=i(moe.get("top_k"), 8),
                dim=i(moe.get("dim"), 32),
                hidden=i(moe.get("hidden"), 64),
                n_layers=i(moe.get("n_layers"), 6),
                load_balance_coef=f(moe.get("load_balance_coef"), 0.01),
                total_params_b=f(moe.get("total_params_b"), 816.0),
                active_params_b=f(moe.get("active_params_b"), 42.0),
            ),
            memory=MemoryConfig(
                bandwidth_gbps=f(memory.get("bandwidth_gbps"), 800.0),
                gpu_pool_gb=f(memory.get("gpu_pool_gb"), 64.0),
                unified_pool_gb=f(memory.get("unified_pool_gb"), 512.0),
                ssd_pool_gb=f(memory.get("ssd_pool_gb"), 1024.0),
                prefetch_depth=i(memory.get("prefetch_depth"), 8),
            ),
            multimodal=MultimodalConfig(
                patch_size=i(mm.get("patch_size"), 8),
                img_size=i(mm.get("img_size"), 32),
                n_audio_frames=i(mm.get("n_audio_frames"), 8),
                n_video_frames=i(mm.get("n_video_frames"), 4),
                dim=i(mm.get("dim"), 32),
            ),
            context=ContextConfig(
                max_tokens=i(ctx.get("max_tokens"), 1048576),
                block_size=i(ctx.get("block_size"), 128),
                gpu_blocks=i(ctx.get("gpu_blocks"), 16),
                offload=b(ctx.get("offload"), True),
            ),
            agent=AgentConfig(
                max_steps=i(agent.get("max_steps"), 12),
                retry_budget=i(agent.get("retry_budget"), 3),
                repo_index_limit=i(agent.get("repo_index_limit"), 256),
                max_files_per_diff=i(agent.get("max_files_per_diff"), 6),
            ),
            model=ModelConfig(
                name=s(model.get("name"), "mock"),
                dim=i(model.get("dim"), 32),
                skill=f(model.get("skill"), 0.5),
            ),
            serving=ServingConfig(
                host=s(serving.get("host"), "127.0.0.1"),
                port=i(serving.get("port"), 8000),
            ),
            seed=i(d.get("seed"), 42),
        )


def load_config(path=None):
    if path is None:
        return Config()
    p = Path(path)
    if not p.exists():
        return Config()
    text = p.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        data = parse(text)
    return Config.from_dict(data)
