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
    parser.add_argument('--segment', type=int, required=True, choices=range(8))
    parser.add_argument('--arch', required=True, choices=['arm', 'x86'])
    parser.add_argument('--jdk', required=True, choices=['25.0.3'])
    parser.add_argument('--host', type=int, required=True, choices=[0])
    args = parser.parse_args()
    root = Path.cwd()
    out = root / 'evidence'
    out.mkdir(exist_ok=True)
    hardware = json.loads((out / 'hardware-admission.json').read_text())
    stratum = next(s for s in PROTOCOL['strata'] if s['id'] == hardware['stratum'])
    pairs_per_segment = stratum['pairs_per_host'] // 8
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
    for relative, expected in PROTOCOL['candidate_source_file_sha256'].items():
        if sha(root / 'candidate' / relative) != expected:
            raise ValueError('Candidate source changed after artifact freeze: ' + relative)
    for qualified, expected in PROTOCOL['candidate_class_bytecode_sha256'].items():
        code = subprocess.check_output(['javap', '-classpath', jars['candidate']['path'], '-p', '-c', qualified])
        if hashlib.sha256(code).hexdigest() != expected:
            raise ValueError('Candidate bytecode changed after artifact freeze: ' + qualified)
    slf4js = list((Path.home() / '.gradle/caches/modules-2/files-2.1/org.slf4j/slf4j-api/2.0.17').glob('*/*.jar'))
    if not slf4js:
        raise ValueError('Missing pinned SLF4J')
    slf4j = slf4js[0]
    classes = out / 'classes'
    classes.mkdir(exist_ok=True)
    classpath = lambda arm: os.pathsep.join([str(classes), jars[arm]['path'], str(slf4j)])
    if args.segment == 0:
        execute(['javac', '--add-modules', 'jdk.incubator.vector', '-cp', classpath('candidate'),
                 '-d', str(classes), str(HERE / 'ConvolutionBoundaryProbe.java')], out / 'compile.log')
        manifest = {'protocol': PROTOCOL, 'harnessRevision': subprocess.check_output(
            ['git', '-C', str(root / 'candidate'), 'rev-parse', 'HEAD'], text=True).strip(),
            'arch': args.arch, 'jdk': args.jdk, 'host': args.host, 'platform': platform.platform(),
            'runId': os.environ.get('GITHUB_RUN_ID'),
            'hardware': json.loads((out / 'hardware-admission.json').read_text()), 'artifacts': jars,
            'java': subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT, text=True),
            'slf4jSha256': sha(slf4j)}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        manifest = json.loads((out / 'manifest.json').read_text())
        if manifest['artifacts'] != jars:
            raise ValueError('Runtime changed between blocks')
    segment = out / f'segment-{args.segment}'
    segment.mkdir(exist_ok=True)
    for pair in range(args.segment * pairs_per_segment, (args.segment + 1) * pairs_per_segment):
        arms = PROTOCOL['arms']
        offset = (pair + args.host) % len(arms)
        order = arms[offset:] + arms[:offset]
        for arm in order:
            command = ['java', '--add-modules', 'jdk.incubator.vector', '-XX:ActiveProcessorCount=4',
                       '-Xms256m', '-Xmx512m', '-cp', classpath(arm),
                       'com.integrallis.vectors.core.ConvolutionBoundaryProbe']
            output = execute(command, segment / f'{pair:03}-{arm}.log')
            row = json.loads(output.splitlines()[-1])
            blocks = [json.loads(line[6:]) for line in output.splitlines() if line.startswith('BLOCK ')]
            if len(blocks) != 5 or any(b['nanos'] < 1_000_000_000 or b['calls'] % 17 for b in blocks):
                raise ValueError('Invalid fixed-duration measurement blocks')
            if row['measured'] != sum(b['calls'] for b in blocks):
                raise ValueError('Measurement count changed')
            row['blocks'] = blocks
            row.update(pair=pair, arm=arm, arch=args.arch, jdk=args.jdk, host=args.host,
                       sha256=jars[arm]['sha256'])
            (segment / f'{pair:03}-{arm}.json').write_text(json.dumps(row, indent=2) + '\n')
            if arm == 'candidate' and (row['coldOracleMismatches'] or row['hotOracleMismatches'] or row['coldHotMismatches']):
                raise ValueError('Candidate numerical regression')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
