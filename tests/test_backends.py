"""Backend registry + mock training signal."""
import unittest

from sparkforge.config import Config
from sparkforge.backends import list_backends, get_backend
from sparkforge.oracle import gold_experts


class TestBackends(unittest.TestCase):
    def test_all_backends_registered(self):
        names = list_backends()
        for expected in ("mock", "cpu", "openai", "vllm", "transformers", "llamacpp"):
            self.assertIn(expected, names)

    def test_mock_trainable(self):
        cfg = Config()
        cfg.model.skill = 0.5
        b = get_backend("mock", cfg)
        self.assertAlmostEqual(b.skill, 0.5)
        b.train_step(0.1)
        self.assertGreater(b.skill, 0.5)

    def test_gold_experts_deterministic(self):
        a = gold_experts("x", 0, 8, 256)
        b = gold_experts("x", 0, 8, 256)
        self.assertEqual(a, b)

    def test_mock_route_skill_effect(self):
        cfg = Config()
        cfg.model.skill = 1.0
        b = get_backend("mock", cfg)
        top = b.route("tok", 0, 8)
        gold = gold_experts("tok", 0, 8, cfg.moe.n_experts)
        # skill 1.0 -> always hits gold
        self.assertEqual(set(top), set(gold))


if __name__ == "__main__":
    unittest.main()
