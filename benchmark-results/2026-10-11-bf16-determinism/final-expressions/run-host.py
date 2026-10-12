import json, pathlib, subprocess, os
out=pathlib.Path('/work/out');java='/opt/jdk/bin/java';commands=[]
def run(label,command,timeout=90,required=True):
 commands.append({'label':label,'command':command});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 with (out/(label+'.log')).open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 commands[-1]['exitCode']=r.returncode;(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 subprocess.run(['aws','s3','cp','/work/out',os.environ['RESULT_URI'],'--recursive','--only-show-errors'],check=True)
 if required and r.returncode:raise RuntimeError(label+' failed')
 return r.returncode==0
common=[java,'--add-modules','jdk.incubator.vector','-XX:ActiveProcessorCount=4','-Xmx512m']
valid=['baseline','rotate']
for arm in ['rotate_pair', 'single_slice', 'sve_lanes']:
 passed=True
 for bits in [64,128,256]:
  passed=run(f'{arm}-cold-{bits}',common+[f'-Dvectors.maxBits={bits}','-cp',arm+'.jar:probes.jar:benchmark.jar','com.integrallis.vectors.core.BFloat16MatrixColdHotProbe','48','5120'],required=False) and passed
 passed=run(arm+'-allocation',common+['-cp',arm+'.jar:probes.jar:benchmark.jar','com.integrallis.vectors.core.BFloat16MatrixAllocationProbe'],required=False) and passed
 if passed:valid.append(arm)
(out/'valid-candidates.json').write_text(json.dumps(valid)+'\n')
order=valid+list(reversed(valid));(out/'order.json').write_text(json.dumps(order)+'\n')
for index,arm in enumerate(order):
 command=[java,'--add-modules','jdk.incubator.vector','-cp',arm+'.jar:benchmark.jar','org.openjdk.jmh.Main','BFloat16MatrixBenchmark.mappedBfloat16.*','-f','1','-wi','5','-i','5','-w','1s','-r','1s','-p','shape=48x5120,4864x896,151936x896','-jvmArgsAppend','-XX:ActiveProcessorCount=4 -Xmx3g','-prof','gc','-rf','json','-rff',str(out/f'{index}-{arm}.json')]
 run(f'{index}-{arm}',command,300)
 rows=json.loads((out/f'{index}-{arm}.json').read_text());assert len(rows)==6 and all(r['primaryMetric']['score']>0 for r in rows)
(out/'STATUS').write_text('BF16_VARIANTS_COMPLETE\n')
