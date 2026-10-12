from pathlib import Path
import subprocess, shutil, hashlib, json, tarfile
repo=Path(__file__).resolve().parents[2];root=repo/'experiments/bf16-tiles'
central=repo.parent/'models-gdn-proof/benchmark-results/2026-10-11-gdn-proof'
version='bf16-tiles-v1';stage=Path('/tmp')/version;stage.mkdir(exist_ok=False)
for arm in ['tile2parallel','tile4parallel','tile4serial']:
 shutil.copy(root/(arm+'.jar'),stage)
for name in ['baseline.jar','benchmark.jar']:
 shutil.copy(Path('/tmp/vectors-bf16-final-expressions-v1')/name,stage)
shutil.copy(root/'probes.jar',stage)
shutil.copy(root/'run-host.py',stage)
shutil.copytree(root,stage/'source',ignore=shutil.ignore_patterns('*.jar','classes','probes','__pycache__'))
shutil.copy(repo/'vectors-core/src/test/java/com/integrallis/vectors/core/BFloat16MatrixColdHotProbe.java',stage/'source')
shutil.copy(repo/'vectors-core/src/test/java/com/integrallis/vectors/core/BFloat16MatrixAllocationProbe.java',stage/'source')
meta=json.loads((central/'payload-bf16-row-layout-v1.json').read_text());meta={k:meta[k] for k in ['jdk','rust']}
meta.update(runName='gdn-proof-'+version,sourceRepository='vectors',revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),maximumSecondsPerHost=3600,maximumCombinedUsd=2,
 files={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.rglob('*')) if p.is_file()})
(stage/'inputs.sha256').write_text(''.join(f'{sha}  {name}\n' for name,sha in meta['files'].items()))
(stage/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
archive=central/('gdn-proof-'+version+'.tar.gz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for p in sorted(stage.iterdir()):tar.add(p,arcname=p.name)
meta.update(payloadUri='s3://models-qual-077051030817/payload/gdn-proof-'+version+'.tar.gz',payloadSha256=hashlib.sha256(archive.read_bytes()).hexdigest())
(central/('payload-'+version+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
subprocess.run(['aws','s3','cp',str(archive),meta['payloadUri'],'--only-show-errors'],check=True)
print(archive,meta['payloadSha256'],flush=True)
