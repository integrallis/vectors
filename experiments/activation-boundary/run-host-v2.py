import hashlib,json,os,pathlib,subprocess,time
out=pathlib.Path('/work/out'); config=json.loads(pathlib.Path('confirmation.json').read_text()); model=config['models'][0]
def status(value):
 (out/'STATUS').write_text(value+'\n');subprocess.run(['aws','s3','cp',str(out),os.environ['RESULT_URI'],'--recursive','--only-show-errors'],check=True)
def run(cmd,name):
 start=time.monotonic()
 with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:r=subprocess.run(cmd,stdout=so,stderr=se,timeout=2400)
 (out/(name+'.execution.json')).write_text(json.dumps({'command':cmd,'exitCode':r.returncode,'wallSeconds':time.monotonic()-start},indent=2));assert r.returncode==0,name
 return [] if name=='jfr-allocation' else [json.loads(l) for l in (out/(name+'.stdout')).read_text().splitlines() if l.startswith('{')]
status('BUILDING');run(['/root/.cargo/bin/cargo','build','--locked','--release','--manifest-path','rust/native/Cargo.toml'],'rust-build')
run(['curl','-fL','--retry','3',model['downloadUri'],'-o','model.gguf'],'download');assert hashlib.file_digest(open('model.gguf','rb'),'sha256').hexdigest()==model['sha256']
base=['/opt/jdk/bin/java','--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xms2g','-Xmx10g']
def execute(arm,phase,name,extra=[],flags=[]):
 return run(['/usr/bin/time','-v',*base,*flags,'-cp',f'lib/{arm}/*:route-proof.jar','com.integrallis.models.backend.purejava.qwen35.ProductionRouteProof',phase,'model.gguf','/work/rust/native/target/release/libjmodels_kernels.so','data/tasks.json',*extra],name)
status('JFR_CONTROL');execute('native','timing','jfr-control',['0'],['-XX:StartFlightRecording=filename=/work/out/control.jfr,settings=profile,dumponexit=true','-XX:FlightRecorderOptions=stackdepth=64'])
run(['/opt/jdk/bin/jfr','print','--json','--events','jdk.ObjectAllocationSample','/work/out/control.jfr'],'jfr-allocation');status('JFR_COMPLETE')
records={}
for arm in ['native','boundary']:
 status('COHORT_'+arm);rows=execute(arm,'cohort',arm+'-cohort');records[arm]={r['id']:r for r in rows if r['kind']=='request'};assert len(records[arm])==148
changes=[]
for key,r in records['native'].items():
 assert len(r['fullLogitSha256'])==9
 for field in ['fullLogitSha256','inputTokens','greedyTokens','prediction']:
  if r[field]!=records['boundary'][key][field]:changes.append({'id':key,'field':field})
(out/'correctness.json').write_text(json.dumps({'requests':148,'changes':changes},indent=2));assert not changes
for pair in range(10):
 for arm in (['native','boundary'] if pair%2==0 else ['boundary','native']):
  status(f'TIMING_{pair}_{arm}');rows=execute(arm,'timing',f'pair{pair}-{arm}',[str(pair)]);requests=[r for r in rows if r['kind']=='request'];assert len(requests)==35;assert [r['repeat'] for r in requests]==list(range(-32,3))
status('COMPLETE')
