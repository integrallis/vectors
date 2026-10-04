"""Contract tests for the independent evidence auditor; no performance work is run."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class AuditTest(unittest.TestCase):
    def fixture(self, root):
        for arm, ns in [('baseline', 1000), ('candidate', 800)]:
            for rep in range(3):
                (root / f'fixture-{arm}-{rep}.log').write_text(''.join(
                    f'RESULT,fixture,{i},{ns},1234,10\n' for i in range(7)))
        (root / 'summary.json').write_text(json.dumps({'fixture': {
            'baseline_us': .1, 'candidate_us': .08, 'less_time_percent': 20.,
            'exact_parity': True,
            'forks_ns_per_query': {'baseline': [100.] * 3, 'candidate': [80.] * 3}}}))

    def audit(self, root):
        return subprocess.run([sys.executable, str(Path(__file__).with_name('audit.py')),
                               str(root)], capture_output=True).returncode

    def test_complete_evidence_passes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            self.assertEqual(0, self.audit(root))

    def test_incomplete_or_tampered_evidence_fails(self):
        for change in ['missing_fork', 'missing_sample', 'duplicate_sample', 'changed_digest',
                       'changed_summary', 'changed_count', 'changed_fork_median']:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                self.fixture(root)
                log = root / 'fixture-candidate-2.log'
                text = log.read_text()
                if change == 'missing_fork':
                    log.unlink()
                elif change == 'missing_sample':
                    log.write_text('\n'.join(text.splitlines()[:-1]))
                elif change == 'duplicate_sample':
                    log.write_text(text + text.splitlines()[0] + '\n')
                elif change == 'changed_digest':
                    log.write_text(text.replace(',1234,', ',1235,'))
                elif change == 'changed_count':
                    log.write_text(text.replace(',10\n', ',11\n'))
                else:
                    summary = root / 'summary.json'
                    data = json.loads(summary.read_text())
                    if change == 'changed_summary':
                        data['fixture']['less_time_percent'] = 21.
                    else:
                        data['fixture']['forks_ns_per_query']['candidate'][0] = 81.
                    summary.write_text(json.dumps(data))
                self.assertNotEqual(0, self.audit(root))


if __name__ == '__main__':
    unittest.main()
