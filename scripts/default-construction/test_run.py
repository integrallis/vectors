import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class OutputIsolationTest(unittest.TestCase):
    def test_output_inside_checkout_is_rejected_without_creating_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = root / 'baseline', root / 'candidate'
            baseline.mkdir()
            candidate.mkdir()
            for checkout in [baseline, candidate]:
                output = checkout / 'measurement-output'
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).with_name('run.py')),
                     str(baseline), str(candidate), str(output),
                     '--generation', str(root / 'source-generation')],
                    capture_output=True, text=True)
                self.assertEqual(2, result.returncode, result.stderr)
                self.assertIn('outside both source checkouts', result.stderr)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
