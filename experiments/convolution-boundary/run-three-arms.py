"""Bounded local kernel diagnostic; actual JAR paths and source revisions are explicit inputs."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess

parser = argparse.ArgumentParser()
for name in ('published', 'broad', 'candidate', 'slf4j', 'classes', 'output', 'java'):
    parser.add_argument('--' + name, required=True)
parser.add_argument('--candidate-source', required=True)
args = parser.parse_args()
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
artifacts = {a: Path(getattr(args, a)).resolve() for a in ('published', 'broad', 'candidate')}
manifest = {'kind': 'local component diagnostic, not release qualification',
            'sources': {'published': 'v0.1.29', 'broad': '6893a587d7f1d8fd2cd58e85310ef87579b42b25',
                        'candidate': args.candidate_source},
            'platform': platform.platform(), 'java': subprocess.check_output([args.java, '-version'], stderr=subprocess.STDOUT, text=True),
            'artifacts': {a: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for a,p in artifacts.items()}}
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
rows = []
for block in range(6):
    order = list(artifacts)
    order = order[block % 3:] + order[:block % 3]
    for arm in order:
        cmd = [args.java, '--add-modules', 'jdk.incubator.vector', '-Xms256m', '-Xmx512m', '-cp',
               os.pathsep.join([args.classes, str(artifacts[arm]), args.slf4j]),
               'com.integrallis.vectors.core.ConvolutionBoundaryProbe']
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
        (output / f'block{block}-{arm}.log').write_text(result.stdout)
        if result.returncode: raise RuntimeError(f'{arm} failed: {result.stdout}')
        row = json.loads(result.stdout.splitlines()[-1]); row.update(block=block, arm=arm, command=cmd)
        rows.append(row)
        (output / 'raw.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(block, arm, row['coldBytes'], row['hotBytesPerCall'], row['hotNanosPerCall'], flush=True)
comparisons = []
for reference in ('published', 'broad'):
    for metric in ('coldBytes', 'coldNanos', 'hotNanosPerCall'):
        logs = [math.log(next(r[metric] for r in rows if r['block']==i and r['arm']=='candidate') /
                         next(r[metric] for r in rows if r['block']==i and r['arm']==reference)) for i in range(6)]
        rng=random.Random(86431)
        draws=sorted(math.exp(statistics.mean(rng.choices(logs,k=6))) for _ in range(20000))
        comparisons.append({'reference': reference, 'metric': metric, 'ratio': math.exp(statistics.mean(logs)),
                            'ci95': [draws[500],draws[19499]]})
summary={'comparisons':comparisons, 'exactOraclePass': all(r['coldOracleMismatches']==0 and r['hotOracleMismatches']==0 for r in rows),
         'hotBytesPerCall': {a:[r['hotBytesPerCall'] for r in rows if r['arm']==a] for a in artifacts}}
(output / 'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
