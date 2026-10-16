"""Run the eval harness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkforge.config import load_config
from sparkforge.backends import get_backend
from sparkforge.eval.routing_eval import routing_recall
from sparkforge.eval.coding_eval import coding_pass_rate
from sparkforge.eval.capacity_eval import capacity_report

cfg = load_config("configs/default.yaml")
backend = get_backend(cfg.model.name, cfg)
print("routing recall:", routing_recall(backend, 24, cfg.moe.top_k, cfg.moe.n_experts, cfg.moe.n_layers))
print("coding pass rate:", coding_pass_rate(cfg.model.skill))
print("capacity:", capacity_report(cfg.moe.total_params_b, cfg.memory.unified_pool_gb))
