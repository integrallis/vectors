"""Run one fixed checkpoint using unchanged public JMH and immutable actual JARs."""
from pathlib import Path
import subprocess,json,hashlib,sys,os,time
here=Path(__file__).resolve().parent;P=json.loads((here/'protocol.json').read_text());payload=Path('payload').resolve();out=Path('evidence').resolve();segment=int(sys.argv[1]);assert 0<=segment<8
hardware=json.loads((out/'hardware-admission.json').read_text());assert hardware['accepted'];spec=next(s for s in P['strata'] if s['id']==hardware['stratum']);slot=hardware['slot'];assert 0<=slot<spec['hosts']
original=json.loads((payload/'protocol.json').read_text())
for name,digest in original['files'].items():assert hashlib.sha256((payload/name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256((payload/'candidate.jar').read_bytes()).hexdigest()==P['candidate_sha256'];assert hashlib.sha256((payload/'baseline.jar').read_bytes()).hexdigest()==P['published_sha256']
assert '25.0.3' in subprocess.check_output(['java','-version'],stderr=subprocess.STDOUT,text=True)
m={'protocol':P,'hardware':hardware,'runId':os.environ['GITHUB_RUN_ID'],'jdk':subprocess.check_output(['java','-version'],stderr=subprocess.STDOUT,text=True),'files':original['files']}
if segment==0:(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
else:assert json.loads((out/'manifest.json').read_text())==m
folder=out/f'segment-{segment}';folder.mkdir(exist_ok=True);assert not list(folder.glob('*-pair*.json'))
def execute(name,cmd):
 start=time.time()
 with (folder/(name+'.log')).open('w') as f:r=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,timeout=180)
 (folder/(name+'.command.json')).write_text(json.dumps({'command':list(map(str,cmd)),'exitCode':r.returncode,'elapsedSeconds':time.time()-start},indent=2)+'\n');assert r.returncode==0,name
classes=out/'classes'
if segment==0:
 classes.mkdir(exist_ok=True)
 execute('compile',['javac','--add-modules','jdk.incubator.vector','-cp',f'{payload}/baseline.jar:{payload}/lib/*','-processorpath',':'.join(map(str,(payload/'lib').glob('*.jar'))),'-d',classes,payload/'BFloat16MatrixBenchmark.java',payload/'BFloat16MatrixColdHotProbe.java'])
 execute('numeric-candidate',['java','--add-modules','jdk.incubator.vector','-XX:ActiveProcessorCount=4','-Xmx1g','-cp',f'{payload}/candidate.jar:{classes}','com.integrallis.vectors.core.BFloat16MatrixColdHotProbe','48','5120'])
cases=list(spec['counts']);cases=cases[slot%len(cases):]+cases[:slot%len(cases)];schedule=[(case,pair) for pair in range(max(spec['counts'].values())) for case in cases if pair<spec['counts'][case]]
for case,pair in schedule[len(schedule)*segment//8:len(schedule)*(segment+1)//8]:
 method,shape=P['cases'][case];ci=list(P['cases']).index(case)
 for arm in (['baseline','candidate'] if (slot+pair+ci)%2==0 else ['candidate','baseline']):
  name=f'{case}-pair{pair:03d}-{arm}';print('START',name,flush=True)
  execute(name,['java','--add-modules','jdk.incubator.vector','-cp',f'{payload}/{arm}.jar:{classes}:{payload}/lib/*','org.openjdk.jmh.Main',f'BFloat16MatrixBenchmark.{method}$','-f','1','-wi','5','-i','5','-w','1s','-r','1s','-p',f'shape={shape}','-jvmArgsAppend','-XX:ActiveProcessorCount=4 -Xmx3g','-prof','gc','-rf','json','-rff',folder/(name+'.json')])
 print('PAIR_COMPLETE',case,pair,flush=True)
(folder/'STATUS').write_text('COMPLETE\n')
