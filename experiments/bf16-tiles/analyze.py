"""Compare six complete counterbalanced process pairs; retain every shape and valid trial."""
import json
import math
import pathlib
import random
import statistics
import sys

folder = pathlib.Path(sys.argv[1])
arms = json.loads((folder / 'candidate-gates.json').read_text())['valid']
samples = {}
for pair in range(6):
    for arm in arms:
        rows = json.loads((folder / f'pair{pair}-{arm}.json').read_text())
        assert len(rows) == 6
        for row in rows:
            key = (row['benchmark'], row['params']['shape'])
            score = row['primaryMetric']['score']
            unit = row['primaryMetric']['scoreUnit']
            factors = {'ns/op': 1, 'us/op': 1000, 'ms/op': 1000000, 's/op': 1000000000}
            assert unit in factors and score > 0
            samples.setdefault(key, {}).setdefault(arm, []).append({
                'pair': pair, 'nsPerOp': score * factors[unit],
                'bytesPerOp': row['secondaryMetrics']['gc.alloc.rate.norm']['score']})
assert len(samples) == 6


def paired(candidate, reference):
    logs = [math.log(c / r) for c, r in zip(candidate, reference)]
    rng = random.Random(716309)
    boot = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs))))
                  for _ in range(20000))
    low, high = boot[500], boot[19499]
    return {'ratio': math.exp(statistics.mean(logs)), 'ci95': [low, high],
            'precisionTargetMet': high - low <= .02,
            'direction': 'slower' if low > 1 else 'faster' if high < 1 else 'unresolved',
            'processRatios': [math.exp(value) for value in logs]}


comparisons = []
for (benchmark, shape), records in sorted(samples.items()):
    assert all([r['pair'] for r in records[arm]] == list(range(6)) for arm in arms)
    for arm in arms[1:]:
        comparisons.append({'benchmark': benchmark, 'shape': shape, 'candidate': arm,
                            'reference': 'baseline',
                            'time': paired([r['nsPerOp'] for r in records[arm]],
                                           [r['nsPerOp'] for r in records['baseline']]),
                            'candidateBytesPerOp': [r['bytesPerOp'] for r in records[arm]],
                            'baselineBytesPerOp': [r['bytesPerOp'] for r in records['baseline']]})
print(json.dumps({'method': 'Geometric mean of process-paired ratios; paired percentile bootstrap, '
                           '20000 resamples, seed 716309; six independent process pairs',
                  'comparisons': comparisons}, indent=2))
