from pathlib import Path
import json,subprocess,os,time,hashlib,math,random,statistics
out=Path('/work/out')
def status(value):
 (out/'STATUS').write_text(value+'\n');subprocess.run(['aws','s3','cp',str(out),os.environ['RESULT_URI'],'--recursive','--only-show-errors'],check=True)
def run(label,args):
 start=time.monotonic()
 with (out/(label+'.stdout')).open('w') as so,(out/(label+'.stderr')).open('w') as se:r=subprocess.run(args,stdout=so,stderr=se,timeout=90)
 (out/(label+'.execution.json')).write_text(json.dumps({'command':args,'exitCode':r.returncode,'wallSeconds':time.monotonic()-start},indent=2)+'\n');assert r.returncode==0,label
 rows=[json.loads(line) for line in (out/(label+'.stdout')).read_text().splitlines() if line.startswith('{')];return next(x['nanos'] for x in rows if x['kind']=='load')
status('DOWNLOADING_PINNED_QWEN')
for entry in json.loads(Path('qwen-files.json').read_text())['files']:
 p=Path('qwen')/entry['path'];p.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['curl','-fsSL','--retry','3',entry['uri'],'-o',str(p)],check=True)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 assert h.hexdigest()==entry['sha256']
base=['/opt/jdk/bin/java','--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xms2g','-Xmx10g']
def command(arm,trace=False):
 return base+(['-Xlog:class+load=info,class+init=info:file=/work/out/trace-'+arm+'.log'] if trace else [])+['-cp','lib/'+arm+'/*:model-proof.jar','com.integrallis.models.backend.purejava.Bf16ModelProof','startup','qwen','.','data/tasks.json','0']
for arm in ['control','candidate']:run('trace-'+arm,command(arm,True))
rows=[];status('TIMING')
for pair in range(100):
 times={}
 for arm in (['control','candidate'] if pair%2==0 else ['candidate','control']):times[arm]=run('pair'+str(pair)+'-'+arm,command(arm))
 rows.append({'pair':pair,**times});(out/'paired-loads.json').write_text(json.dumps(rows,indent=2)+'\n')
 if pair%10==9:status('TIMING_PAIR_'+str(pair))
logs=[math.log(r['candidate']/r['control']) for r in rows];rng=random.Random(716309);samples=sorted(math.exp(statistics.mean(rng.choices(logs,k=100))) for _ in range(20000));lo,hi=samples[500],samples[19499]
(out/'analysis.json').write_text(json.dumps({'pairs':100,'ratio':math.exp(statistics.mean(logs)),'ci95':[lo,hi],'direction':'slower' if lo>1 else 'faster' if hi<1 else 'unresolved','precisionTargetMet':hi-lo<=.02,'method':'100 prospective counterbalanced fresh-process pairs; geometric mean, paired percentile bootstrap20000 seed716309; previous failed ten-pair result retained separately','rows':rows},indent=2)+'\n');status('COMPLETE')
