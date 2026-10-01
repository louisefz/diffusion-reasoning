import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from tracking import Tracker, scalar_metrics


class TrackingTests(unittest.TestCase):
    def test_scalar_names(self):
        values = scalar_metrics({"step": 10, "loss": 0.5,
            "validation": {"compose/d8": {"accuracy": 0.25, "total": 4, "correct": 1}}})
        self.assertEqual(values["train/step"], 10)
        self.assertEqual(values["train/loss"], 0.5)
        self.assertEqual(values["validation/compose/d8/accuracy"], 0.25)

    def test_disabled(self):
        with Tracker(SimpleNamespace(wandb_mode="disabled"), {}, 0) as tracker:
            tracker.log({"step": 1, "loss": 0.5})
        self.assertIsNone(tracker.run)

    def test_logging_failure_does_not_stop_training(self):
        tracker = Tracker(SimpleNamespace(wandb_mode="disabled"), {}, 0)
        tracker.run = Mock()
        tracker.run.log.side_effect = RuntimeError("network")
        with self.assertWarns(UserWarning):
            tracker.log({"step": 1})
        tracker.log({"step": 2})
        self.assertEqual(tracker.run.log.call_count, 1)
        tracker.__exit__(None, None, None)
        tracker.run.finish.assert_called_once_with(exit_code=1)

    def test_exception_not_swallowed(self):
        tracker = Tracker(SimpleNamespace(wandb_mode="disabled"), {}, 0)
        tracker.run = Mock()
        self.assertFalse(tracker.__exit__(ValueError, ValueError(), None))
        tracker.run.finish.assert_called_once_with(exit_code=1)
