"""Run one prospectively assigned checkpoint; never retry or change counts from results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from protocol import *

parser = argparse.ArgumentParser()
parser.add_argument('candidate', type=Path)
parser.add_argument('baseline', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--arch', choices=HOSTS, required=True)
parser.add_argument('--host', type=int, default=0)
parser.add_argument('--segment', type=int, required=True)
args = parser.parse_args()
args.output.mkdir(exist_ok=True, parents=True)
segment = args.output / f'segment-{args.segment}'
segment.mkdir(exist_ok=True)
if list(segment.glob('*-pair*-*.json')):
    raise RuntimeError('Refusing to overwrite or repeat a checkpoint')
commands = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(label, command, timeout=180):
    receipt = {'label': label, 'command': list(map(str, command)), 'startedUnix': time.time()}
    commands.append(receipt)
    try:
        with (segment / f'{label}.log').open('w') as log:
            process = subprocess.run(receipt['command'], stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        receipt['exitCode'] = process.returncode
        if process.returncode: raise RuntimeError(f'{label}: exit{process.returncode}')
    except Exception as error:
        receipt['error'] = repr(error)
        raise
    finally:
        receipt['elapsedSeconds'] = time.time() - receipt['startedUnix']
        (segment / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')


def core(root):
    jars = [p.resolve() for p in (root / 'vectors-core/build/libs').glob('vectors-core-*.jar') if not p.name.endswith(('-sources.jar', '-javadoc.jar'))]
    assert len(jars) == 1, jars
    return jars[0]


jars = {'baseline': core(args.baseline), 'candidate': core(args.candidate)}
benchmark = Path('vectors-bench/src/jmh/java/com/integrallis/vectors/bench/BFloat16MatrixBenchmark.java')
assert (args.candidate / benchmark).read_bytes() == (args.baseline / benchmark).read_bytes(), 'Benchmark changed'
manifest = {'arch': args.arch, 'hostIndex': args.host, 'hostCount': HOSTS[args.arch],
            'hostPairCounts': host_counts(args.arch), 'cpuCount': os.cpu_count(),
            'jdk': subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT).decode(),
            'benchmarkSha256': digest(args.baseline / benchmark), 'artifacts': {},
            'schedule': schedule(args.arch, args.host), 'segments': SEGMENTS,
            'protocolSha256': digest(Path(__file__).with_name('protocol.py'))}
assert manifest['cpuCount'] >= 4, 'Four available CPUs required'
assert '25.0.3' in manifest['jdk'], 'JDK version changed'
for arm, jar in jars.items():
    actual = digest(jar)
    assert actual == ARTIFACTS[arm], (arm, actual, ARTIFACTS[arm])
    root = getattr(args, arm)
    manifest['artifacts'][arm] = {'sha256': actual, 'jar': str(jar), 'source': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD']).decode().strip()}
manifest_path = args.output / 'manifest.json'
if manifest_path.exists():
    assert json.loads(manifest_path.read_text()) == manifest, 'Host/artifact state changed between checkpoints'
else:
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
libs = args.output / 'jmh-libs'
libs.mkdir(exist_ok=True)
dependencies = [
    ('org/openjdk/jmh', 'jmh-core', '1.37', 'dc0eaf2bbf0036a70b60798c785d6e03a9daf06b68b8edb0f1ba9eb3421baeb3'),
    ('org/openjdk/jmh', 'jmh-generator-annprocess', '1.37', '6a5604b5b804e0daca1145df1077609321687734a8b49387e49f10557c186c77'),
    ('net/sf/jopt-simple', 'jopt-simple', '5.0.4', 'df26cc58f235f477db07f753ba5a3ab243ebe5789d9f89ecf68dd62ea9a66c28'),
    ('org/apache/commons', 'commons-math3', '3.6.1', '1e56d7b058d28b65abd256b8458e3885b674c1d588fa43cd7d1cbb9c7ef2b308'),
]
for group, name, version, expected in dependencies:
    path = libs / f'{name}-{version}.jar'
    if not path.exists(): urllib.request.urlretrieve(f'https://repo.maven.apache.org/maven2/{group}/{name}/{version}/{path.name}', path)
    assert digest(path) == expected, path
classes = args.output / 'jmh-classes'
if args.segment == 0:
    classes.mkdir(exist_ok=True)
    run('compile-unchanged-benchmark', ['javac', '--add-modules', 'jdk.incubator.vector', '-cp',
        f'{jars["baseline"]}{os.pathsep}{libs}/*', '-processorpath', os.pathsep.join(str(p) for p in sorted(libs.glob('*.jar'))), '-d', classes, args.baseline / benchmark])
else:
    assert classes.exists(), 'First checkpoint did not compile benchmark'
for pair in segment_schedule(args.arch, args.host, args.segment):
    method, shape = CASES[pair['case']]
    for arm in pair['order']:
        label = f'{pair["case"]}-pair{pair["pair"]:03d}-{arm}'
        print(f'START arch={args.arch} host={args.host} segment={args.segment} {label}', flush=True)
        run(label, ['java', '--add-modules', 'jdk.incubator.vector', '-cp',
            f'{jars[arm]}{os.pathsep}{classes}{os.pathsep}{libs}/*', 'org.openjdk.jmh.Main',
            f'BFloat16MatrixBenchmark.{method}$', '-f', '1', '-wi', '5', '-i', '5', '-w', '1s', '-r', '1s',
            '-p', f'shape={shape}', '-jvmArgsAppend', '-XX:ActiveProcessorCount=4 -Xmx3g -XX:+UnlockDiagnosticVMOptions -XX:+LogCompilation '
            + '-XX:LogFile=' + str((segment / (label + '-hotspot-%p.xml')).resolve())
            + ' -XX:+PrintCompilation -XX:CompileCommand=print,com.integrallis.vectors.core.BFloat16Matrix::*',
            '-prof', 'gc', '-rf', 'json', '-rff', segment / f'{label}.json'])
        records = json.loads((segment / f'{label}.json').read_text())
        assert len(records) == 1, records
        record = records[0]
        assert record['benchmark'].endswith('.' + method) and record['params']['shape'] == shape
        assert record['primaryMetric']['score'] > 0 and math.isfinite(record['primaryMetric']['score'])
        assert record['primaryMetric']['scoreUnit'] == 'us/op'
    print(f'PAIRED_COMPLETE arch={args.arch} host={args.host} {pair["case"]} pair={pair["pair"]}', flush=True)
(segment / 'STATUS').write_text('COMPLETE\n')
print(f'CHECKPOINT_COMPLETE arch={args.arch} host={args.host} segment={args.segment}', flush=True)
