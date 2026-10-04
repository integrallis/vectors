from pathlib import Path
import tarfile,hashlib,json
root=Path('/opt/query')
runs=[p for p in root.iterdir() if p.is_dir() and (p/'summary.json').exists()]
with tarfile.open(root/'review-evidence.tar.gz','w:gz') as t:
 for run in sorted(runs):
  for p in sorted(run.iterdir()):
   if p.is_file() and p.suffix in {'.json','.log','.java','.py'}:t.add(p,arcname=str(Path(run.name)/p.name))
 for p in sorted(root.iterdir()):
  if p.is_file() and p.suffix in {'.log','.txt','.json','.py','.patch'}:t.add(p,arcname='qualification/'+p.name)
 for p in (root/'source').glob('*/build/test-results/test/TEST-*.xml'):
  t.add(p,arcname='qualification/test-results/'+p.parent.parent.parent.parent.name+'/'+p.name)
with tarfile.open(root/'replay-assets.tar.gz','w:gz') as t:
 for p in sorted(root.iterdir()):
  if p.is_dir() and (p.name.startswith('candidate-') or p.name=='public-glove-baseline'):
   t.add(p,arcname=p.name)
 for p in [root/'review-evidence.tar.gz',root/'host.txt']:t.add(p,arcname=p.name)
record={}
for name in ['review-evidence.tar.gz','replay-assets.tar.gz']:
 p=root/name
 with p.open('rb') as f:record[name]={'bytes':p.stat().st_size,'sha256':hashlib.file_digest(f,'sha256').hexdigest()}
(root/'evidence-archives.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
