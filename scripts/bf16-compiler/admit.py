"""Admit hardware before building or timing; unmatched hardware remains explicitly uncovered."""
import json
import os
from pathlib import Path
import subprocess
import sys
arch = sys.argv[1]
raw = subprocess.check_output(['lscpu', '-J'], text=True)
fields = {r['field'].rstrip(':'): r['data'] for r in json.loads(raw)['lscpu']}
cpu = fields['Model name']
expected = 'AMD EPYC 9V45 96-Core Processor' if arch == 'x86' else 'Neoverse-N2'
accepted = cpu == expected
out = Path('evidence'); out.mkdir(exist_ok=True)
(out / 'hardware-admission.json').write_text(json.dumps({'arch': arch, 'cpu': cpu,
    'expectedCpu': expected, 'accepted': accepted, 'lscpu': fields,
    'scope': 'Admission by hardware identity before measurements. No latency-based selection.'}, indent=2) + '\n')
with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
    stream.write('accepted=' + str(accepted).lower() + '\n')
print('ADMITTED' if accepted else 'HARDWARE_UNAVAILABLE_NO_TIMING', cpu)
