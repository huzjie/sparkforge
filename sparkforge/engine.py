"""SparkEngine: top-level facade tying model + memory + context + agent."""
from .config import Config, load_config
from .backends import get_backend


class SparkEngine:
    """One object to run inference (with unified-memory scheduling) and coding
    tasks (with the agent loop).
    """

    def __init__(self, config: Config):
        self.config = config
        self.backend = get_backend(config.model.name, config)

    def complete(self, prompt, max_tokens=16, media_emb=None):
        return self.backend.generate(prompt, max_tokens)

    def code(self, task, repo_root, symbols=None):
        from .agent.coding_loop import CodingAgent
        agent = CodingAgent(self.config, backend=self.backend)
        return agent.run(task, repo_root, symbols)

    def memory_report(self):
        from .memory.tiers import MemoryTiers
        m = self.config.memory
        tiers = MemoryTiers(m.gpu_pool_gb, m.unified_pool_gb, m.ssd_pool_gb)
        return {"pools": tiers.pools, "used": tiers.used}
