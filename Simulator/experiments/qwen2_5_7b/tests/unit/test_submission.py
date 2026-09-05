"""Regression for the final launch being absent from a phase's trace snapshot."""
from types import SimpleNamespace
import unittest

from Simulator.experiments.qwen2_5_7b.submission import phase_kernel_ids


class SubmissionTests(unittest.TestCase):
    def test_includes_pending_last_kernel_without_device_sync(self):
        prefix = "# header\nLAUNCH_KERNEL,0,0,0,old,attr,0\n"
        session = SimpleNamespace(trace_log=prefix + "LAUNCH_KERNEL,1,0,0,new,attr,0\n")
        callbacks = [lambda: setattr(session, "trace_log", session.trace_log + "LAUNCH_KERNEL,2,0,0,last,attr,0\n")]

        def enqueue_marker(marker):
            callbacks.append(marker)
            for callback in callbacks:
                callback()

        self.assertEqual(phase_kernel_ids(session, len(prefix), enqueue_marker), [1, 2])
        self.assertNotIn("DEVICE_SYNC", session.trace_log)

    def test_does_not_return_partial_ids_if_stream_does_not_drain(self):
        with self.assertRaisesRegex(TimeoutError, "Host kernel submissions"):
            phase_kernel_ids(SimpleNamespace(trace_log=""), 0, lambda callback: None, timeout=0)


if __name__ == "__main__":
    unittest.main()
