import unittest

from Simulator.experiments.qwen2_5_7b.analysis.trace_summary import activity_state_cycles, analyze, check_run_status, merge, overlap


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
    def test_exploratory_analysis_never_accepts_an_execution_failure(self):
        check_run_status({"status": "passed"})
        result = {"status": "completed_with_numerical_mismatch",
                  "arguments": {"allow_numerical_mismatch": True},
                  "probe": {"numerical_status": "failed"}}
        with self.assertRaises(ValueError):
            check_run_status(result)
        check_run_status(result, True)
        with self.assertRaises(ValueError):
            check_run_status({**result, "status": "failed"}, True)
        with self.assertRaises(ValueError):
            check_run_status({**result, "arguments": {}}, True)

    def test_interval_union_and_clipping(self):
        intervals = merge([(4, 8), (1, 5), (8, 10), (12, 15)])
        self.assertEqual(intervals, [[1, 10], [12, 15]])
        self.assertEqual(overlap(intervals, 3, 13), 8)

    def test_simultaneous_states_are_exact_and_clip_phase_boundaries(self):
        occupied = {"VPU": [(1, 5)], "MXU0": [(2, 6)], "MXU1": [(3, 4)]}
        counts = activity_state_cycles(occupied, 0, 7)
        self.assertEqual(counts, {"none": 2, "VPU": 1, "MXU0": 1, "VPU+MXU0": 2,
                                  "MXU1": 0, "VPU+MXU1": 0, "MXU0+MXU1": 0,
                                  "VPU+MXU0+MXU1": 1})
        clipped = activity_state_cycles(occupied, 3, 5)
        self.assertEqual(clipped["VPU+MXU0+MXU1"], 1)
        self.assertEqual(clipped["VPU+MXU0"], 1)
        self.assertEqual(sum(clipped.values()), 2)

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
