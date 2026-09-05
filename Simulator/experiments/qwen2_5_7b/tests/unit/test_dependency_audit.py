import unittest

from Simulator.experiments.qwen2_5_7b.analysis.dependency_audit import audit


def fixture():
    launches = [{"kernel_id": 0, "attribute": "/x/runtime_0000/attribute/0"},
                {"kernel_id": 1, "attribute": "/y/runtime_0000/attribute/0"}]
    log = '\n'.join([
        '[LoadConfig] Loaded configuration file "/x/runtime_0000/attribute/0"',
        '[TOGParser] Store Node arg0 Numa_id: 0:',
        '[LoadConfig] Loaded configuration file "/y/runtime_0000/attribute/0"',
        '[TOGParser] Load Node arg0 Numa_id: 0:',
        '[3][Core 0][INST_ISSUED ][INST_ID=0] MOVOUT (addr_name=arg0 async=false)',
        '[5][Core 0][INST_FINISHED ][INST_ID=0] MOVOUT (addr_name=arg0)',
        '[10][Core 0][DRAM_RESP_DONE ][INST_ID=0] MOVOUT (addr_name=arg0)',
        'Kernel 0 has completed - operation: producer finished at cycle 11',
        'Kernel 0 execution summary - Started at: 1 cycles',
        '[13][Core 0][INST_ISSUED ][INST_ID=1] MOVIN (addr_name=arg0 async=false)',
        '[16][Core 0][INST_FINISHED ][INST_ID=1] MOVIN (addr_name=arg0)',
        'Kernel 1 has completed - operation: consumer finished at cycle 17',
        'Kernel 1 execution summary - Started at: 12 cycles',
    ])
    return log, launches


class DependencyAuditTests(unittest.TestCase):
    def test_complete_ordered_trace(self):
        report, _ = audit(*fixture())
        self.assertEqual(report['errors'], [])
        self.assertEqual(report['store_response_events_checked'], 1)

    def test_reject_early_kernel_completion(self):
        log, launches = fixture()
        report, _ = audit(log.replace('finished at cycle 11', 'finished at cycle 4'), launches)
        self.assertTrue(any('before last event' in error for error in report['errors']))

    def test_reject_early_consumer(self):
        log, launches = fixture()
        report, _ = audit(log.replace('[13][Core', '[6][Core').replace('Started at: 12', 'Started at: 5'), launches)
        self.assertTrue(any('before kernel 0' in error for error in report['errors']))

    def test_store_injection_is_not_memory_completion(self):
        log, launches = fixture()
        report, _ = audit('\n'.join(line for line in log.splitlines() if 'DRAM_RESP_DONE' not in line), launches)
        self.assertTrue(any('missing DRAM_RESP_DONE' in error for error in report['errors']))

    def test_reject_opcode_misattribution(self):
        log, launches = fixture()
        with self.assertRaises(ValueError):
            audit(log.replace('Store Node', 'Load Node'), launches)


if __name__ == '__main__':
    unittest.main()
