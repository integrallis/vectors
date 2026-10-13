#!/usr/bin/env python3
"""Fixed two-arm screens; retain every process, including failures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(command, output, timeout=240):
    started = time.time()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as stream:
        try:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=timeout, check=False)
            status = result.returncode
        except subprocess.TimeoutExpired:
            status = 'TIMEOUT'
    receipt = {'command': command, 'status': status, 'elapsedSeconds': time.time() - started}
    output.with_suffix('.command.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if status != 0:
        raise RuntimeError(f'{output}: {status}')
    return output.read_text()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment', type=int, required=True, choices=range(5))
    parser.add_argument('--arch', required=True, choices=['arm', 'x86'])
    parser.add_argument('--jdk', required=True, choices=['25.0.3'])
    parser.add_argument('--host', type=int, required=True, choices=[0])
    args = parser.parse_args()
    root = Path.cwd()
    out = root / 'evidence'
    out.mkdir(exist_ok=True)
    jars = {}
    for arm in PROTOCOL['arms']:
        matches = list((root / arm / 'vectors-core/build/libs').glob('vectors-core-*.jar'))
        matches = [p for p in matches if not p.name.endswith(('-sources.jar', '-javadoc.jar'))]
        if len(matches) != 1:
            raise ValueError(f'{arm}: expected one runtime jar, got {matches}')
        jar = matches[0]
        jars[arm] = {'path': str(jar), 'sha256': sha(jar), 'revision': subprocess.check_output(
            ['git', '-C', str(root / arm), 'rev-parse', 'HEAD'], text=True).strip()}
        if jars[arm]['sha256'] != PROTOCOL[arm + '_sha256'] or (
                arm == 'published' and jars[arm]['revision'] != PROTOCOL[arm + '_revision']):
            raise ValueError(f'{arm} provenance mismatch: {jars[arm]}')
    slf4js = list((Path.home() / '.gradle/caches/modules-2/files-2.1/org.slf4j/slf4j-api/2.0.17').glob('*/*.jar'))
    if not slf4js:
        raise ValueError('Missing pinned SLF4J')
    slf4j = slf4js[0]
    classes = out / 'classes'
    classes.mkdir(exist_ok=True)
    classpath = lambda arm: os.pathsep.join([str(classes), jars[arm]['path'], str(slf4j)])
    if args.segment == 0:
        execute(['javac', '--add-modules', 'jdk.incubator.vector', '-cp', classpath('candidate'),
                 '-d', str(classes), str(HERE / 'ConvolutionBoundaryProbe.java'),
                 str(HERE / 'CompiledSiluProof.java')], out / 'compile.log')
        manifest = {'protocol': PROTOCOL, 'harnessRevision': subprocess.check_output(
            ['git', '-C', str(root / 'candidate'), 'rev-parse', 'HEAD'], text=True).strip(),
            'arch': args.arch, 'jdk': args.jdk, 'host': args.host, 'platform': platform.platform(),
            'runId': os.environ.get('GITHUB_RUN_ID'), 'artifacts': jars,
            'java': subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT, text=True),
            'slf4jSha256': sha(slf4j)}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        manifest = json.loads((out / 'manifest.json').read_text())
        if manifest['artifacts'] != jars:
            raise ValueError('Runtime changed between blocks')
    if args.segment == 4:
        java = ['java', '--add-modules', 'jdk.incubator.vector', '-XX:ActiveProcessorCount=4',
                '-Xms256m', '-Xmx512m']
        # Proof follows all timing pairs and runs in independent processes.
        for fma in [True, False]:
            options = [] if fma else ['-Dvectors.useVectorFMA=false', '-Dvectors.useScalarFMA=false']
            execute(java + options + ['-cp', classpath('candidate'),
                    'com.integrallis.vectors.core.CompiledSiluProof',
                    'com.integrallis.vectors.core.ReproducibleConvolutionSilu', 'apply', 'exhaustive'],
                    out / f'proof/exhaustive-{fma}.log', timeout=600)
        return
    segment = out / f'segment-{args.segment}'
    segment.mkdir(exist_ok=True)
    for pair in [args.segment]:
        arms = PROTOCOL['arms']
        offset = (pair + args.host) % len(arms)
        order = arms[offset:] + arms[:offset]
        for arm in order:
            command = ['java', '--add-modules', 'jdk.incubator.vector', '-XX:ActiveProcessorCount=4',
                       '-Xms256m', '-Xmx512m', '-cp', classpath(arm),
                       'com.integrallis.vectors.core.ConvolutionBoundaryProbe']
            output = execute(command, segment / f'{pair:03}-{arm}.log')
            row = json.loads(output.splitlines()[-1])
            row.update(pair=pair, arm=arm, arch=args.arch, jdk=args.jdk, host=args.host,
                       sha256=jars[arm]['sha256'])
            (segment / f'{pair:03}-{arm}.json').write_text(json.dumps(row, indent=2) + '\n')
            if arm == 'candidate' and (row['coldOracleMismatches'] or row['hotOracleMismatches'] or row['coldHotMismatches']):
                raise ValueError('Candidate numerical regression')
            print(json.dumps(row), flush=True)
        if pair == 0:
            measured = {arm: json.loads((segment / f'{pair:03}-{arm}.json').read_text())
                        for arm in PROTOCOL['arms']}
            candidate, published = measured['candidate'], measured['published']
            checks = {
                'warmLatency': candidate['hotNanosPerCall'] <= published['hotNanosPerCall'] * PROTOCOL['latency_upper_limit'],
                'coldAllocation': candidate['coldBytes'] <= published['coldBytes'] * PROTOCOL['cold_allocation_individual_limit'],
                'warmAllocation': candidate['hotBytesPerCall'] <= published['hotBytesPerCall'] + PROTOCOL['warm_allocation_absolute_tolerance_bytes'],
            }
            (out / 'first-pair-screen.json').write_text(json.dumps(checks, indent=2) + '\n')
            if not all(checks.values()):
                raise ValueError('First paired screen rejected; retained both arms, no qualification claim')


if __name__ == '__main__':
    main()
