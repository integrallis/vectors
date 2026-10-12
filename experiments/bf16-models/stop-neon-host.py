import json,subprocess,time,datetime
from pathlib import Path
base=Path('/Users/briansam-bodden/Code/java-ai/projects/models-gdn-proof/benchmark-results/2026-10-11-gdn-proof')
receipt=json.loads((base/'hosts-bf16-models-v3.json').read_text())
pending={h['arch']:h for h in receipt['hosts']}
assert {h['instanceId'] for h in pending.values()}=={'i-074982f87b258b12b'}
while pending:
 for arch,h in list(pending.items()):
  uri=h['resultUri'];tail='kernel-matrix.execution.json'
  q=subprocess.run(['aws','s3','cp',uri+tail,'-','--only-show-errors'],capture_output=True,text=True)
  if q.returncode:continue
  execution=json.loads(q.stdout)
  assert execution['exitCode']==0,execution
  folder=base/'remote-bf16-models-v3'/arch
  subprocess.run(['aws','s3','sync',uri,str(folder),'--only-show-errors'],check=True)
  for pair in range(12):
   for arm in ['baseline','tile2parallel']:
    rows=json.loads((folder/f'pair{pair}-{arm}.json').read_text())
    assert len(rows)==6 and all(r['primaryMetric']['score']>0 for r in rows)
  result=subprocess.run(['aws','ec2','terminate-instances','--region','us-east-1','--instance-ids',h['instanceId']],check=True,capture_output=True,text=True)
  record={'instanceId':h['instanceId'],'phase':'bf16-models-v3','reason':'NEON real-model fixtures and all12 kernel pairs complete; full macros remain unmeasured within declared one-hour cap','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'terminationRequest':json.loads(result.stdout)}
  (base/f'{arch}-bf16-models-v3-kernel-stop.json').write_text(json.dumps(record,indent=2)+'\n')
  print(json.dumps(record),flush=True);del pending[arch]
 if pending:time.sleep(30)
print('BOTH_KERNEL_HOSTS_STOP_REQUESTED',flush=True)
