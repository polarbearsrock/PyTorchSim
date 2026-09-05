import unittest

from Simulator.experiments.qwen2_5_7b.analysis.trace_summary import analyze, merge, overlap


def event(cycle, action, identity, kind, latency, hidden):
    return f"[{cycle}][Core 0][{action} ][INST_ID={identity}] COMP (compute_type={kind} compute_cycle={latency} overlapping_cycle={hidden})"


def sample_log():
    return "\n".join([
        event(1, "INST_ISSUED", 0, 0, 3, 0),
        event(2, "INST_ISSUED", 1, 2, 5, 2),
        event(3, "INST_ISSUED", 2, 1, 6, 1),
        event(4, "INST_ISSUED", 3, 1, 5, 2),
        event(4, "INST_FINISHED", 0, 0, 3, 0),
        "Core [0] : Total_cycles: 6",
        event(7, "INST_FINISHED", 1, 2, 5, 2),
        event(9, "INST_FINISHED", 2, 1, 6, 1),
        event(10, "INST_FINISHED", 3, 1, 5, 2),
        "Systolic array [0] utilization(%): 0.00, active_cycles: 7",
        "Systolic array [1] utilization(%): 0.00, active_cycles: 5",
        "Vector unit utilization(%): 0.00, active cycle: 4",
        "Core [0] : Total_cycles: 12",
        "Total execution cycles: 12",
    ])


class TraceTests(unittest.TestCase):
    def test_interval_union_and_clipping(self):
        intervals = merge([(4, 8), (1, 5), (8, 10), (12, 15)])
        self.assertEqual(intervals, [[1, 10], [12, 15]])
        self.assertEqual(overlap(intervals, 3, 13), 8)

    def test_queue_assignment_and_native_counter_replay(self):
        jobs, occupied, _, total, native = analyze(sample_log())
        self.assertEqual([j["unit"] for j in jobs], ["VPU", "MXU0", "MXU1", "MXU0"])
        self.assertEqual(occupied["MXU0"], [[2, 10]])
        # Reset at cycle 6 makes a later bubble subtraction saturate at zero.
        # This intentionally differs from simply subtracting all bubbles once.
        self.assertEqual(native, {"MXU0": 7, "MXU1": 5, "VPU": 4})
        self.assertEqual(total, 12)

    def test_reject_incomplete_or_inconsistent_trace(self):
        with self.assertRaises(AssertionError):
            analyze(sample_log().replace(event(10, "INST_FINISHED", 3, 1, 5, 2), ""))
        with self.assertRaises(AssertionError):
            analyze(sample_log().replace(event(10, "INST_FINISHED", 3, 1, 5, 2), event(11, "INST_FINISHED", 3, 1, 5, 2)))
        with self.assertRaises(AssertionError):
            analyze(sample_log().replace("active_cycles: 7", "active_cycles: 8"))


if __name__ == "__main__":
    unittest.main()
