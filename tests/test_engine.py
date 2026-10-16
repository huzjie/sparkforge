"""Engine facade + generate."""
import unittest

from sparkforge.config import Config
from sparkforge.engine import SparkEngine
from sparkforge.model.spark import SparkModel
from sparkforge.model.generate import generate


class TestEngine(unittest.TestCase):
    def test_complete(self):
        eng = SparkEngine(Config())
        out = eng.complete("hello")
        self.assertIsInstance(out, str)

    def test_forward(self):
        model = SparkModel(Config())
        out, aux, activated = model.forward("def f(): pass")
        self.assertEqual(len(activated), Config().moe.n_layers)

    def test_generate_deterministic(self):
        model = SparkModel(Config())
        a = generate(model, "hello", 4)
        b = generate(model, "hello", 4)
        self.assertEqual(a, b)

    def test_memory_report(self):
        eng = SparkEngine(Config())
        rep = eng.memory_report()
        self.assertIn("pools", rep)


if __name__ == "__main__":
    unittest.main()
