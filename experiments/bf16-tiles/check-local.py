from pathlib import Path
import subprocess,json,time
root=Path(__file__).resolve().parent
java='/Users/briansam-bodden/.sdkman/candidates/java/25.0.3-tem/bin/java'
base=Path('/tmp/vectors-bf16-final-expressions-v1');commands=[]
for arm in ['tile2parallel','tile4parallel','tile4serial']:
 cp=f'{root}/{arm}.jar:{base}/probes.jar:{base}/benchmark.jar'
 for label,args in [('cold',['com.integrallis.vectors.core.BFloat16MatrixColdHotProbe','48','5120']),('alloc',['com.integrallis.vectors.core.BFloat16MatrixAllocationProbe'])]:
  command=[java,'--add-modules','jdk.incubator.vector','-XX:ActiveProcessorCount=4','-Xmx512m','-cp',cp]+args
  with (root/f'{arm}-{label}.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
  commands.append({'arm':arm,'kind':label,'exitCode':r.returncode,'command':command});print(arm,label,r.returncode,flush=True)
  if r.returncode:raise SystemExit(1)
(root/'local-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
