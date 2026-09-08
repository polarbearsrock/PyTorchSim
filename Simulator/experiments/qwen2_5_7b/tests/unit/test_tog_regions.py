"""Regression-auditor controls; the real compiler tests live in integration/."""
import unittest

from Simulator.experiments.qwen2_5_7b.tests.integration.tog_regions import audit_regions, check_graph_regions


def region(kind, body):
    return (f'.insn r CUSTOM_3, 0, 0x40 compute_type = "{kind}"\n'
            + body + '\n.insn r CUSTOM_3, 0, 0x41\n')


PUSH = '"vcix.iv"() <{opcode = 0 : i64}>'
READ = '"vcix.v.i"() <{opcode = 2 : i64}>'
STORE = '"vector.transfer_write"() {test.role = "accum-0"}'


class TogRegionTests(unittest.TestCase):
    def test_counts_unequal_transfers(self):
        regions = audit_regions(region("MatmulCompute", "\n".join([PUSH] * 8 + [READ, STORE] * 16)))
        self.assertEqual((regions[0]["pushes"], regions[0]["reads"], regions[0]["stores"]), (8, 16, 16))

    def test_rejects_old_vpu_tail(self):
        with self.assertRaisesRegex(ValueError, "outside MatmulCompute"):
            audit_regions(region("VectorCompute", READ))
        with self.assertRaisesRegex(ValueError, "outside MatmulCompute"):
            audit_regions(READ)

    def test_requires_last_accumulator_write_and_separate_epilogue(self):
        with self.assertRaisesRegex(ValueError, "all accumulator writes"):
            audit_regions(region("MatmulCompute", PUSH + "\n" + READ))
        tail = '"arith.negf"() {test.role = "epilogue-0"}'
        with self.assertRaisesRegex(ValueError, "expected VectorCompute"):
            audit_regions(region("MatmulCompute", "\n".join([PUSH, READ, STORE, tail])))

    def test_rejects_bad_markers(self):
        for text in ('.insn r CUSTOM_3, 0, 0x41',
                     '.insn r CUSTOM_3, 0, 0x40',
                     '.insn r CUSTOM_3, 0, 0x40 compute_type = "VectorCompute"'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                audit_regions(text)

    def test_graph_labels_are_independent_of_marker_labels(self):
        graph = {0: {"node_name": "root", "children": [1]},
                 1: {"node_name": "ComputeNode", "compute_type": 1, "children": []}}
        regions = audit_regions(region("MatmulCompute", "\n".join([PUSH, READ, STORE])))
        check_graph_regions(f"graph = {graph!r}", regions)
        graph[1]["compute_type"] = 0
        with self.assertRaisesRegex(ValueError, "Graph unit labels disagree"):
            check_graph_regions(f"graph = {graph!r}", regions)


if __name__ == "__main__":
    unittest.main()
