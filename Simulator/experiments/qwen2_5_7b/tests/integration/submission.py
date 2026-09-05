"""Check the metadata fence against the actual OpenReg host stream."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from Simulator.experiments.qwen2_5_7b.submission import phase_kernel_ids


class HostSubmissionTests(unittest.TestCase):
    def test_openreg_marker_captures_all_launches_without_device_sync(self):
        stream = torch.npu.default_stream()
        session = SimpleNamespace(trace_log="")
        for identity in range(3):
            def submit(identity=identity):
                session.trace_log += f"LAUNCH_KERNEL,{identity},0,0,graph,attribute,0\n"
            stream.launch_kernel(submit)
        with patch.object(torch.npu, "synchronize", side_effect=AssertionError("Must not add a DEVICE_SYNC")):
            self.assertEqual(phase_kernel_ids(session, 0, stream.launch_kernel), [0, 1, 2])
        self.assertNotIn("DEVICE_SYNC", session.trace_log)


if __name__ == "__main__":
    unittest.main()
