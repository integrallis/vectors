"""Frozen twelve-pair BF16 comparison of actual core JARs on one host."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import urllib.request

candidate_root, baseline_root, output = map(lambda p: Path(p).resolve(), sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
commands = []

def run(label, argv, timeout=600):
    started = time.monotonic()
    entry = {'label': label, 'argv': list(map(str, argv))}
    commands.append(entry)
    try:
        with (output / (label + '.log')).open('w') as log:
            process = subprocess.run(entry['argv'], stdout=log, stderr=subprocess.STDOUT,
                                     timeout=timeout)
        entry['exitCode'] = process.returncode
        if process.returncode:
            raise RuntimeError(f'{label} failed: {process.returncode}')
    except Exception as error:
        entry['error'] = repr(error)
        raise
    finally:
        entry['seconds'] = time.monotonic() - started
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def core(root):
    jars = list((root / 'vectors-core/build/libs').glob('vectors-core-*.jar'))
    jars = [p for p in jars if not p.name.endswith(('-sources.jar', '-javadoc.jar'))]
    assert len(jars) == 1, jars
    return jars[0]

jars = {'baseline': core(baseline_root), 'candidate': core(candidate_root)}
manifest = {'pairs': 12, 'jdk': subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT).decode(),
            'cpuCount': os.cpu_count(), 'artifacts': {}}
for arm, root in [('baseline', baseline_root), ('candidate', candidate_root)]:
    manifest['artifacts'][arm] = {'source': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD']).decode().strip(),
                                 'jar': str(jars[arm]), 'sha256': digest(jars[arm])}
assert manifest['cpuCount'] >= 4, 'Four real available CPUs required by frozen protocol'
libs = output / 'jmh-libs'
libs.mkdir(exist_ok=True)
dependencies = [
    ('org/openjdk/jmh', 'jmh-core', '1.37', 'dc0eaf2bbf0036a70b60798c785d6e03a9daf06b68b8edb0f1ba9eb3421baeb3'),
    ('org/openjdk/jmh', 'jmh-generator-annprocess', '1.37', '6a5604b5b804e0daca1145df1077609321687734a8b49387e49f10557c186c77'),
    ('net/sf/jopt-simple', 'jopt-simple', '5.0.4', 'df26cc58f235f477db07f753ba5a3ab243ebe5789d9f89ecf68dd62ea9a66c28'),
    ('org/apache/commons', 'commons-math3', '3.6.1', '1e56d7b058d28b65abd256b8458e3885b674c1d588fa43cd7d1cbb9c7ef2b308'),
]
for group, name, version, expected in dependencies:
    path = libs / f'{name}-{version}.jar'
    urllib.request.urlretrieve(f'https://repo.maven.apache.org/maven2/{group}/{name}/{version}/{path.name}', path)
    assert digest(path) == expected, path
manifest['jmhDependencies'] = {p.name: digest(p) for p in sorted(libs.glob('*.jar'))}
source = Path('vectors-bench/src/jmh/java/com/integrallis/vectors/bench/BFloat16MatrixBenchmark.java')
assert (candidate_root / source).read_bytes() == (baseline_root / source).read_bytes(), 'Benchmark changed'
manifest['benchmarkSha256'] = digest(baseline_root / source)
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
classes = output / 'jmh-classes'
classes.mkdir(exist_ok=True)
run('compile-unchanged-benchmark', ['javac', '--add-modules', 'jdk.incubator.vector', '-cp',
    f'{jars["baseline"]}{os.pathsep}{libs}/*', '-processorpath', os.pathsep.join(str(p) for p in sorted(libs.glob('*.jar'))), '-d', classes, baseline_root / source])
for pair in range(12):
    order = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
    for arm in order:
        label = f'pair{pair}-{arm}'
        (output / 'STATUS').write_text(label + '\n')
        run(label, ['java', '--add-modules', 'jdk.incubator.vector', '-cp',
            f'{jars[arm]}{os.pathsep}{classes}{os.pathsep}{libs}/*', 'org.openjdk.jmh.Main',
            'BFloat16MatrixBenchmark.mappedBfloat16.*', '-f', '1', '-wi', '5', '-i', '5',
            '-w', '1s', '-r', '1s', '-p', 'shape=48x5120,4864x896,151936x896',
            '-jvmArgsAppend', '-XX:ActiveProcessorCount=4 -Xmx3g', '-prof', 'gc', '-rf', 'json',
            '-rff', output / (label + '.json')])
        records = json.loads((output / (label + '.json')).read_text())
        assert len(records) == 6 and all(r['primaryMetric']['score'] > 0 for r in records)

samples = {}
for pair in range(12):
    for arm in jars:
        for row in json.loads((output / f'pair{pair}-{arm}.json').read_text()):
            assert row['primaryMetric']['scoreUnit'] == 'us/op'
            key = (row['benchmark'], row['params']['shape'])
            samples.setdefault(key, {}).setdefault(arm, []).append({
                'pair': pair, 'usPerOp': row['primaryMetric']['score'],
                'bytesPerOp': row['secondaryMetrics']['gc.alloc.rate.norm']['score']})
assert len(samples) == 6
comparisons = []
for (benchmark, shape), arms in sorted(samples.items()):
    assert all([r['pair'] for r in arms[a]] == list(range(12)) for a in jars)
    ratios = [c['usPerOp'] / b['usPerOp'] for c, b in zip(arms['candidate'], arms['baseline'])]
    logs = list(map(math.log, ratios))
    rng = random.Random(716309)
    bootstrap = sorted(math.exp(statistics.mean(rng.choices(logs, k=12))) for _ in range(20000))
    low, high = bootstrap[500], bootstrap[19499]
    comparisons.append({'benchmark': benchmark, 'shape': shape,
        'time': {'ratio': math.exp(statistics.mean(logs)), 'ci95': [low, high],
                 'direction': 'slower' if low > 1 else 'faster' if high < 1 else 'unresolved',
                 'precisionTargetMet': high - low <= .02, 'processRatios': ratios},
        'allocationBytesPerOp': {a: [r['bytesPerOp'] for r in arms[a]] for a in jars}})
passed = all(c['time']['direction'] == 'faster' and c['time']['precisionTargetMet'] for c in comparisons)
(output / 'analysis.json').write_text(json.dumps({'kernelLatencyGate': 'PASS' if passed else 'NOT_PASSED',
    'wholeModelQualification': 'REQUIRED_SEPARATELY', 'comparisons': comparisons}, indent=2) + '\n')
(output / 'STATUS').write_text('COMPLETE\n')
if not passed:
    raise SystemExit('At least one kernel latency/precision gate did not pass; see retained analysis')
