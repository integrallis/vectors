import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import analyze


class ProtocolTests(unittest.TestCase):
    def test_hierarchical_constant_ratio(self):
        import math
        with patch.dict(analyze.PROTOCOL, bootstrap_samples=100):
            result = analyze.interval({0: [math.log(.8)]*32, 1: [math.log(.8)]*32})
        self.assertAlmostEqual(result['estimate'], .8)
        self.assertEqual(result['ci95'][0], result['ci95'][1])

    def fixture(self, root, cold=80):
        for arch in ['x86', 'arm']:
            for jdk in ['25', '25.0.3']:
                for host in range(2):
                    d = root / f'{arch}-{jdk}-{host}'
                    d.mkdir()
                    artifacts = {arm: {'sha256': analyze.PROTOCOL.get(arm+'_sha256', 'candidate-hash')}
                                 for arm in analyze.PROTOCOL['arms']}
                    (d/'manifest.json').write_text(json.dumps(dict(protocol=analyze.PROTOCOL,
                        arch=arch, jdk=jdk, host=host, artifacts=artifacts,
                        harnessRevision='frozen-source', runId='one-run')))
                    segment = d/'segment-0'; segment.mkdir()
                    for pair in range(32):
                        for arm in analyze.PROTOCOL['arms']:
                            row = dict(arch=arch, jdk=jdk, host=host, pair=pair, arm=arm,
                                channels=10240, kernel=4, warmup=10000, measured=20000, bits=256, fma=True,
                                sha256=artifacts[arm]['sha256'], coldBytes=cold if arm=='candidate' else 100,
                                coldNanos=100, hotNanosPerCall=80 if arm=='candidate' else 100,
                                hotBytesPerCall=0, coldOracleMismatches=0, hotOracleMismatches=0)
                            (segment/f'{pair}-{arm}.json').write_text(json.dumps(row))

    def test_complete_cohort_and_reject_missing(self):
        with tempfile.TemporaryDirectory() as name, patch.dict(analyze.PROTOCOL, bootstrap_samples=40):
            root = Path(name); self.fixture(root)
            self.assertEqual(analyze.analyze(root)['verdict'], 'PASSED')
            next(root.rglob('0-candidate.json')).unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete cohort'):
                analyze.analyze(root)

    def test_allocation_regression_cannot_pass(self):
        with tempfile.TemporaryDirectory() as name, patch.dict(analyze.PROTOCOL, bootstrap_samples=40):
            root = Path(name); self.fixture(root, cold=101)
            result = analyze.analyze(root)
            self.assertEqual(result['verdict'], 'NOT_PASSED')
            self.assertTrue(all(not r['gates']['coldAllocation'] for r in result['results']))

    def test_mismatched_workload_and_mode_rejected(self):
        with tempfile.TemporaryDirectory() as name, patch.dict(analyze.PROTOCOL, bootstrap_samples=40):
            root = Path(name); self.fixture(root)
            path = next(root.rglob('0-candidate.json'))
            row = json.loads(path.read_text()); row['measured'] = 100
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'Workload mismatch'):
                analyze.analyze(root)
            row['measured'] = 20000; row['fma'] = False
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'SIMD configuration mismatch'):
                analyze.analyze(root)

    def test_numerical_error_cannot_pass(self):
        with tempfile.TemporaryDirectory() as name, patch.dict(analyze.PROTOCOL, bootstrap_samples=40):
            root = Path(name); self.fixture(root)
            path = next(root.rglob('0-candidate.json'))
            row = json.loads(path.read_text()); row['hotOracleMismatches'] = 1
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'numerical mismatch'):
                analyze.analyze(root)


if __name__ == '__main__':
    unittest.main()
