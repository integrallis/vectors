import math
import random
import statistics
SEED = 716309
RESAMPLES = 20_000

def hierarchical_interval(values_by_host, *, resamples=RESAMPLES, seed=SEED, exponentiate=True):
    """Equal-weight hosts; resample hosts then independent paired observations within each draw.

    Inputs are log(candidate/baseline) for latency, paired differences for allocation.
    Resampling a host twice independently resamples its paired observations twice.
    No grouping by measured speed or exclusion of a compilation mode is permitted.
    """
    hosts = sorted(values_by_host)
    if not hosts or any(not values_by_host[h] for h in hosts):
        raise ValueError('empty host or cohort')
    if any(not math.isfinite(v) for h in hosts for v in values_by_host[h]):
        raise ValueError('non-finite observation')
    means = [statistics.mean(values_by_host[h]) for h in hosts]
    transform = math.exp if exponentiate else lambda v: v
    estimate = transform(statistics.mean(means))
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sampled_hosts = rng.choices(hosts, k=len(hosts))
        host_means = []
        for host in sampled_hosts:
            values = values_by_host[host]
            host_means.append(statistics.mean(rng.choices(values, k=len(values))))
        draws.append(transform(statistics.mean(host_means)))
    draws.sort()
    return {'estimate': estimate, 'ci95': [draws[int(resamples * .025)], draws[int(resamples * .975) - 1]]}


def latency_verdict(interval):
    low, high = interval['ci95']
    direction = 'slower' if low > 1 else 'faster' if high < 1 else 'unresolved'
    return {**interval, 'direction': direction, 'precisionTargetMet': high - low <= .02}

