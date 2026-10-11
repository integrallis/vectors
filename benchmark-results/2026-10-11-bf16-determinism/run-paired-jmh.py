#!/usr/bin/env python3
"""Paired BF16 measurements with explicit baseline/fixed core jars on one host."""
import argparse, hashlib, json, pathlib, platform, subprocess
p = argparse.ArgumentParser()
p.add_argument('--java', required=True)
p.add_argument('--baseline', type=pathlib.Path, required=True)
p.add_argument('--fixed', type=pathlib.Path, required=True)
p.add_argument('--benchmark', type=pathlib.Path, required=True)
p.add_argument('--output', type=pathlib.Path, required=True)
p.add_argument('--profile-gc', action='store_true')
p.add_argument('--warmups', type=int, default=3)
p.add_argument('--iterations', type=int, default=3)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
meta = {'host': platform.platform(), 'java': subprocess.check_output([a.java, '-version'], stderr=subprocess.STDOUT, text=True),
        'jars': {k: {'path': str(getattr(a,k).resolve()), 'sha256': sha(getattr(a,k))} for k in ['baseline','fixed','benchmark']},
        'order': ['baseline','fixed','fixed','baseline'], 'commands': []}
for index, arm in enumerate(meta['order']):
    command = [a.java, '--add-modules', 'jdk.incubator.vector', '-cp', str(getattr(a,arm).resolve()) + ':' + str(a.benchmark.resolve()),
               'org.openjdk.jmh.Main', 'BFloat16MatrixBenchmark.mappedBfloat16.*', '-f', '1', '-wi', str(a.warmups), '-i', str(a.iterations),
               '-w', '1s', '-r', '1s', '-p', 'shape=48x5120,4864x896,151936x896',
               '-jvmArgsAppend', '-XX:ActiveProcessorCount=4 -Xmx3g', '-rf', 'json', '-rff', str(a.output.resolve()/f'{index}-{arm}.json')]
    if a.profile_gc: command += ['-prof', 'gc']
    meta['commands'].append(command)
    (a.output/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    with (a.output/f'{index}-{arm}.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
