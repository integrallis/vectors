#!/usr/bin/env python3
"""Complete-cohort analysis with frozen host/pair bootstrap."""
import json
import math
from pathlib import Path
import random
import statistics
import sys

PROTOCOL = json.loads((Path(__file__).with_name('protocol.json')).read_text())


def interval(groups, log=True):
    hosts = sorted(groups)
    transform = math.exp if log else lambda x: x
    estimate = transform(statistics.mean(statistics.mean(groups[h]) for h in hosts))
    rng = random.Random(PROTOCOL['seed'])
    draws = []
    for _ in range(PROTOCOL['bootstrap_samples']):
        samples = [statistics.mean(rng.choices(groups[h], k=len(groups[h])))
                   for h in rng.choices(hosts, k=len(hosts))]
        draws.append(transform(statistics.mean(samples)))
    draws.sort()
    return {'estimate': estimate, 'ci95': [draws[len(draws)//40], draws[len(draws)*39//40-1]]}


def analyze(root):
    rows = {}
    manifests = {}
    for path in root.rglob('manifest.json'):
        manifest = json.loads(path.read_text())
        key = (manifest['arch'], manifest['jdk'], manifest['host'])
        if key in manifests and manifests[key] != manifest:
            raise ValueError('Conflicting repeated manifest')
        manifests[key] = manifest
    expected_hosts = {(arch, jdk, host) for arch in ['x86', 'arm']
                      for jdk in ['25', '25.0.3'] for host in range(2)}
    if set(manifests) != expected_hosts:
        raise ValueError('Incomplete host cohort')
    candidate_hashes, harnesses, runs = set(), set(), set()
    for key, manifest in manifests.items():
        if manifest['protocol'] != PROTOCOL:
            raise ValueError('Protocol mismatch')
        candidate_hashes.add(manifest['artifacts']['candidate']['sha256'])
        harnesses.add(manifest['harnessRevision'])
        runs.add(manifest['runId'])
        for arm in ['published', 'regressed']:
            if manifest['artifacts'][arm]['sha256'] != PROTOCOL[arm+'_sha256']:
                raise ValueError('Wrong baseline artifact')
    if len(candidate_hashes) != 1 or len(harnesses) != 1 or len(runs) != 1:
        raise ValueError('Mixed provenance')
    for path in root.rglob('*.json'):
        if path.name.endswith('.command.json') or path.parent.name[:8] != 'segment-':
            continue
        row = json.loads(path.read_text())
        key = (row['arch'], row['jdk'], row['host'], row['pair'], row['arm'])
        if key in rows:
            raise ValueError('Duplicate observation')
        rows[key] = row
        for field, expected_value in [('channels', PROTOCOL['channels']), ('kernel', PROTOCOL['kernel']),
                                      ('warmup', PROTOCOL['warmup_calls']), ('measured', PROTOCOL['measured_calls'])]:
            if row[field] != expected_value:
                raise ValueError('Workload mismatch: ' + field)
        if row['bits'] not in [64,128,256,512] or not isinstance(row['fma'], bool):
            raise ValueError('Invalid SIMD configuration')
        if row['sha256'] != manifests[key[:3]]['artifacts'][key[-1]]['sha256']:
            raise ValueError('Row artifact mismatch')
        for metric in ['coldBytes', 'coldNanos', 'hotBytesPerCall', 'hotNanosPerCall']:
            if not math.isfinite(row[metric]) or row[metric] < 0:
                raise ValueError('Invalid measurement')
        if row['arm'] == 'candidate' and (row['coldOracleMismatches'] or row['hotOracleMismatches']):
            raise ValueError('Candidate numerical mismatch')
    expected = {(*host, pair, arm) for host in expected_hosts
                for pair in range(32) for arm in PROTOCOL['arms']}
    if set(rows) != expected:
        raise ValueError(f'Incomplete cohort: {len(expected-set(rows))} missing')
    for host in expected_hosts:
        for pair in range(32):
            modes = {(rows[(*host,pair,arm)]['bits'], rows[(*host,pair,arm)]['fma'])
                     for arm in PROTOCOL['arms']}
            if len(modes) != 1:
                raise ValueError('Paired SIMD configuration mismatch')
    results = []
    for arch in ['x86', 'arm']:
        for jdk in ['25.0.3', '25']:
            result = {'arch': arch, 'jdk': jdk, 'comparisons': {}}
            for control in ['published', 'regressed']:
                comparisons = {}
                for metric in ['coldBytes', 'coldNanos', 'hotNanosPerCall', 'hotBytesPerCall']:
                    groups = {}
                    for host in range(2):
                        groups[host] = []
                        for pair in range(32):
                            a = rows[(arch, jdk, host, pair, 'candidate')][metric]
                            b = rows[(arch, jdk, host, pair, control)][metric]
                            groups[host].append(a-b if metric == 'hotBytesPerCall' else math.log(a/b))
                    comparisons[metric] = interval(groups, log=metric != 'hotBytesPerCall')
                    comparisons[metric]['largestPair'] = (max(v for g in groups.values() for v in g)
                        if metric == 'hotBytesPerCall' else math.exp(max(v for g in groups.values() for v in g)))
                result['comparisons'][control] = comparisons
            published = result['comparisons']['published']
            lo, hi = published['hotNanosPerCall']['ci95']
            result['gates'] = {
                'warmLatency': hi <= PROTOCOL['latency_upper_limit'],
                'warmLatencyPrecision': hi-lo <= PROTOCOL['latency_interval_width_limit'],
                'coldAllocation': published['coldBytes']['ci95'][1] <= PROTOCOL['cold_allocation_upper_limit'],
                'everyColdAllocationPair': published['coldBytes']['largestPair'] <= PROTOCOL['cold_allocation_individual_limit'],
                'warmAllocation': published['hotBytesPerCall']['ci95'][1] <= PROTOCOL['warm_allocation_absolute_tolerance_bytes'],
            }
            result['passed'] = all(result['gates'].values())
            results.append(result)
    return {'verdict': 'PASSED' if all(r['passed'] for r in results) else 'NOT_PASSED',
            'candidateSha256': next(iter(candidate_hashes)), 'harnessRevision': next(iter(harnesses)),
            'runId': next(iter(runs)), 'processes': len(rows), 'results': results}


if __name__ == '__main__':
    destination = Path(sys.argv[2]); destination.mkdir(parents=True, exist_ok=True)
    try:
        result = analyze(Path(sys.argv[1]))
    except Exception as error:
        (destination/'analysis-error.json').write_text(json.dumps({'error': str(error)}, indent=2)+'\n')
        raise
    (destination/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['verdict'] == 'PASSED' else 1)
