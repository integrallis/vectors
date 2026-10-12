"""Collect only the owned immutable final phases and preserve complete host-paired blocks."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

repo = Path(__file__).resolve().parents[2]
central = repo.parent / 'models-gdn-proof/benchmark-results/2026-10-11-gdn-proof'
phases = {'bf16-models-v9': range(4), 'bf16-models-v10': range(4, 7), 'bf16-models-v11': range(7, 10)}
models = ['qwen-bf16', 'bonsai2']
final_sha = '696c2a58843181a2b02c91043208b56c9c6ebdc976e0d3f44dc051cc1f36d555'
for arch in ['x86', 'arm']:
    aggregate = central / 'bf16-final-combined' / arch
    aggregate.mkdir(parents=True, exist_ok=True)
    hosts = {model: {} for model in models}
    provenance = []
    for phase, pairs in phases.items():
        receipt = central / f'hosts-{phase}-{arch}.json'
        if not receipt.exists():
            print(phase, arch, 'NOT_LAUNCHED')
            continue
        fleet = json.loads(receipt.read_text())
        payload = json.loads((central / f'payload-{phase}.json').read_text())
        assert fleet['payloadSha256'] == payload['payloadSha256']
        assert payload['files']['lib/candidate/vectors-core-0.1.30.jar'] == final_sha
        assert len(fleet['hosts']) == 1
        host = fleet['hosts'][0]
        assert host['arch'] == arch
        raw = central / f'remote-{phase}' / arch
        raw.mkdir(parents=True, exist_ok=True)
        subprocess.run(['aws', 's3', 'sync', host['resultUri'], str(raw), '--only-show-errors'], check=True)
        if (raw / 'provenance.json').exists():
            remote = json.loads((raw / 'provenance.json').read_text())
            assert remote['files'] == payload['files']
        status = (raw / 'STATUS').read_text().strip() if (raw / 'STATUS').exists() else 'BOOTSTRAP'
        print(phase, arch, host['instanceId'], status)
        for model in models:
            for pair in pairs:
                paths = [raw / f'{model}-pair{pair}-{arm}.{suffix}'
                         for arm in ['control', 'candidate', 'published']
                         for suffix in ['stdout', 'stderr', 'execution.json']]
                if not all(p.is_file() for p in paths):
                    continue
                for path in paths:
                    if path.name.endswith('.execution.json'):
                        assert json.loads(path.read_text())['exitCode'] == 0, path
                for path in paths:
                    shutil.copy2(path, aggregate / path.name)
                    provenance.append({'phase': phase, 'host': host['instanceId'],
                                       'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                hosts[model][str(pair)] = host['instanceId']
    (aggregate / 'pair-hosts.json').write_text(json.dumps(hosts, indent=2) + '\n')
    (aggregate / 'aggregation-provenance.json').write_text(json.dumps({
        'runtimeJarSha256': final_sha, 'files': provenance}, indent=2) + '\n')
    for model in models:
        if len(hosts[model]) == 10:
            output = repo / f'experiments/bf16-models/combined-final-{arch}-{model}-analysis.json'
            subprocess.run([sys.executable, str(repo / 'experiments/bf16-models/analyze.py'),
                            str(aggregate), str(output), model], check=True)
