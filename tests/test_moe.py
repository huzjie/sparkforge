"""MoE layer / router correctness."""
import unittest

from sparkforge.config import Config
from sparkforge.moe.layer import MoELayer
from sparkforge.moe.router import Router


class TestMoE(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.layer = MoELayer(16, 4, 16, 32)

    def test_forward_shape(self):
        from sparkforge.tensor import Tensor
        x = Tensor([[0.1] * 16])
        out, aux, top = self.layer.forward(x)
        self.assertEqual(out.shape, (1, 16))
        self.assertEqual(len(top), 4)

    def test_active_ratio(self):
        self.assertAlmostEqual(self.layer.active_ratio(), 4 / 16)

    def test_router_topk(self):
        router = Router(16, 16, 4)
        x = __import__("sparkforge.tensor", fromlist=["Tensor"]).Tensor([[0.0] * 16])
        top, gates, aux = router.route(x)
        self.assertEqual(len(top), 4)
        self.assertEqual(len(set(top)), 4)
        self.assertGreaterEqual(aux, 0.0)

    def test_expert_determinism(self):
        from sparkforge.tensor import Tensor
        x = Tensor([[0.5] * 16])
        a = self.layer.experts[0].forward(x)
        b = self.layer.experts[0].forward(x)
        self.assertEqual(a.data, b.data)


if __name__ == "__main__":
    unittest.main()
