import json,subprocess,time,datetime
from pathlib import Path
base=Path('/Users/briansam-bodden/Code/java-ai/projects/models-gdn-proof/benchmark-results/2026-10-11-gdn-proof')
pending={h['arch']:h for h in json.loads((base/'hosts-bf16-models-v2.json').read_text())['hosts']}
assert {h['instanceId'] for h in pending.values()}=={'i-0f20dd70e58d02955','i-055244d0474d2468b'}
while pending:
 for arch,h in list(pending.items()):
  folder=base/'remote-bf16-models-v2'/arch
  subprocess.run(['aws','s3','sync',h['resultUri'],str(folder),'--only-show-errors'],check=True)
  expected=[folder/f'bonsai2-pair{pair}-{arm}.execution.json' for pair in range(6) for arm in ['published','control','candidate']]
  if not all(p.exists() for p in expected):continue
  for p in expected:
   assert json.loads(p.read_text())['exitCode']==0
   rows=[json.loads(s) for s in p.with_suffix('').with_suffix('.stdout').read_text().splitlines() if s.startswith('{')]
   assert len([r for r in rows if r['kind']=='request'])==7
  result=subprocess.run(['aws','ec2','terminate-instances','--region','us-east-1','--instance-ids',h['instanceId']],check=True,capture_output=True,text=True)
  record={'instanceId':h['instanceId'],'phase':'bf16-models-v2','reason':'All six declared original-host Bonsai pairs complete; pairs6-9 continue in immutable v4/v5 host strata','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'terminationRequest':json.loads(result.stdout)}
  (base/f'{arch}-bf16-models-v2-six-pair-stop.json').write_text(json.dumps(record,indent=2)+'\n')
  print(json.dumps(record),flush=True);del pending[arch]
 if pending:time.sleep(45)
