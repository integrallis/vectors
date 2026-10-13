"""Fixed compiler diagnosis, never a release performance cohort."""
import math
HOSTS = {'x86': 1, 'arm': 1}
SEGMENTS = 8
ARTIFACTS = {
    'baseline': '21eeae4fcaa56fa26ed45d99610142f794cdec0e1d852894e637263f5bd0b40c',
    'candidate': '9403f34da7b0713d3e9fb1c4d507a3f11d745fe1a7b3f4620d8ad24be4c82592',
}
CASES = {
    'matvec_ordinary': ('mappedBfloat16MatVec', '4864x896'),
    'matvec_small': ('mappedBfloat16MatVec', '48x5120'),
}

def host_counts(arch):
    return {'matvec_ordinary' if arch == 'arm' else 'matvec_small': 32}

def schedule(arch, host):
    if host != 0:
        raise ValueError('one declared host per architecture')
    case = next(iter(host_counts(arch)))
    return [{'case': case, 'pair': pair,
             'order': ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']}
            for pair in range(32)]

def segment_schedule(arch, host, segment):
    if not 0 <= segment < SEGMENTS:
        raise ValueError('undeclared checkpoint')
    return schedule(arch, host)[segment * 4:(segment + 1) * 4]
