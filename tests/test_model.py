import unittest
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class ModelTests(unittest.TestCase):
    def setUp(self):
        from model import Config, FlowModel, batch
        from task import generate
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = FlowModel(Config(width=32, layers=1, heads=2, self_condition=True))
        self.c, self.y = batch(generate(2, 8, 4, [2], 1, set(), "a"), self.model.config, "cpu")

    def test_finite_training_gradients(self):
        from model import losses
        loss, _, _ = losses(self.model, self.c, self.y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
        self.assertIsNotNone(self.model.unembed.weight.grad)

    def test_exact_state_replay(self):
        from model import sample
        self.model.eval()
        logits, trace = sample(self.model, self.c, 8, record=True)
        state = trace[3]
        resumed, _ = sample(self.model, self.c, 8, z=state["z"], previous=state["previous"], start_step=3)
        torch.testing.assert_close(logits, resumed, rtol=0, atol=0)
        self.assertEqual(len(trace), 9)
        with self.assertRaises(ValueError):
            sample(self.model, self.c, 8, z=state["z"], start_step=3)

    def test_new_decode_omits_history(self):
        from model import sample
        self.model.eval()
        self.model.config.decode_history = False
        logits, trace = sample(self.model, self.c, 4, record=True)
        with torch.no_grad():
            expected = self.model(self.c, trace[-1]["z"], torch.ones(len(self.y)), mode=1)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    def test_direct_classifier(self):
        from model import direct_logits, losses
        from unittest.mock import patch
        with patch("model.torch.randn", side_effect=AssertionError("No noise needed")):
            a = direct_logits(self.model, self.c)
            b = direct_logits(self.model, self.c)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            loss, fm, ce = losses(self.model, self.c, self.y, "direct")
        self.assertEqual(fm.item(), 0)
        torch.testing.assert_close(loss.detach(), ce)
        loss.backward()
        self.assertIsNotNone(self.model.condition_in.weight.grad)


if __name__ == "__main__":
    unittest.main()
