from pathlib import Path
import hashlib,json,os,subprocess,time,urllib.parse
out=Path('/work/out');config=json.loads(Path('model-proof.json').read_text());failures=[]

def status(value):
 (out/'STATUS').write_text(value+'\n')
 subprocess.run(['aws','s3','cp',str(out),os.environ['RESULT_URI'],'--recursive','--only-show-errors'],check=True)

def run(command,label,timeout=6000,gate=True):
 start=time.monotonic()
 with (out/(label+'.stdout')).open('w') as so,(out/(label+'.stderr')).open('w') as se:
  r=subprocess.run(command,stdout=so,stderr=se,timeout=timeout)
 (out/(label+'.execution.json')).write_text(json.dumps({'command':command,'exitCode':r.returncode,'wallSeconds':time.monotonic()-start},indent=2)+'\n')
 if gate:assert r.returncode==0,label
 return r.returncode

def verify_origin(label, arm):
 rows=[json.loads(line) for line in (out/(label+'.stdout')).read_text().splitlines() if line.startswith('{')]
 origins=[r for r in rows if r['kind']=='runtime-origin'];assert len(origins)==1
 artifacts=list(Path('lib',arm).glob('vectors-core-*.jar'));assert len(artifacts)==1
 loaded=Path(urllib.parse.urlsplit(origins[0]['bf16']).path).resolve()
 assert loaded==artifacts[0].resolve(),(loaded,artifacts)

def fetch(entry,directory,label):
 path=directory/entry['path'];path.parent.mkdir(parents=True,exist_ok=True)
 run(['curl','-fL','--retry','3',entry['uri'],'-o',str(path)],'download-'+label+'-'+entry['path'].replace('/','_'))
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 assert h.hexdigest()==entry['sha256'],entry['path']
 if 'sizeBytes' in entry:assert path.stat().st_size==entry['sizeBytes'],entry['path']

status('KERNEL_GATES')
run(['python3','run-kernel-host.py'],'kernel-matrix')
status('BUILDING')
for arm in ['published','corrected']:
 run(['/root/.cargo/bin/cargo','build','--locked','--release','--manifest-path',f'rust/{arm}/Cargo.toml'],'build-'+arm)
 path=Path(f'rust/{arm}/target/release/libjmodels_kernels.so')
 (out/('native-'+arm+'.sha256')).write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
base=['/opt/jdk/bin/java','--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xms2g','-Xmx10g']
assert not any(a.startswith(('-Dmodels.','-Dvectors.')) for a in base)
for model in config['models']:
 key=model['key'];status('DOWNLOAD_'+key)
 for entry in model['files']:fetch(entry,Path(key),key)
 path=str(Path(key)/model['entrypoint'])
 for arm in config['arms']:
  status('FIXTURE_'+key+'_'+arm)
  native='published' if arm=='published' else 'corrected'
  command=[*base,'-cp',f'lib/{arm}/*:model-proof.jar','com.integrallis.models.backend.purejava.Bf16ModelProof','fixture',path,f'/work/rust/{native}/target/release/libjmodels_kernels.so']
  code=run(command,key+'-'+arm+'-fixture',gate=False)
  verify_origin(key+'-'+arm+'-fixture',arm)
  if code:failures.append({'model':key,'arm':arm,'gate':'model-fixture','exitCode':code})
  (out/'fixture-gates.json').write_text(json.dumps(failures,indent=2)+'\n')
 if any(f['model']==key and f['arm'] in ['control','candidate'] for f in failures):
  status('TIMING_BLOCKED_BY_FIXTURE_'+key)
  continue
 for pair in range(10):
  arms=config['arms'];order=arms[pair%len(arms):]+arms[:pair%len(arms)]
  if pair%2:order.reverse()
  for arm in order:
   status(f'TIMING_{key}_{pair}_{arm}')
   native='published' if arm=='published' else 'corrected'
   command=['/usr/bin/time','-v',*base,'-cp',f'lib/{arm}/*:model-proof.jar','com.integrallis.models.backend.purejava.Bf16ModelProof','timing',path,f'/work/rust/{native}/target/release/libjmodels_kernels.so','data/tasks.json',str(pair)]
   run(command,f'{key}-pair{pair}-{arm}')
   verify_origin(f'{key}-pair{pair}-{arm}',arm)
status('COMPLETE_WITH_FAILED_FIXTURES' if failures else 'COMPLETE')
