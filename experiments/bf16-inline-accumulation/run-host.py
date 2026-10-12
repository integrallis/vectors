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

status('BUILDING')
run(['/root/.cargo/bin/cargo','build','--locked','--release','--manifest-path','rust/corrected/Cargo.toml'],'build-native')
base=['/opt/jdk/bin/java','--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xms2g','-Xmx10g']
model=next(m for m in config['models'] if m['key']=='bonsai2')
for entry in model['files']:fetch(entry,Path('bonsai2'),'bonsai2')
path=str(Path('bonsai2')/model['entrypoint'])
for pair,arms in enumerate([['inline','control'],['control','inline']]):
 for arm in arms:
  label=f'bonsai2-pair{pair}-{arm}';status(label)
  run(['/usr/bin/time','-v',*base,'-cp',f'lib/{arm}/*:model-proof.jar','com.integrallis.models.backend.purejava.Bf16ModelProof','timing',path,'/work/rust/corrected/target/release/libjmodels_kernels.so','data/tasks.json',str(pair)],label,timeout=360)
  verify_origin(label,arm)
status('COMPLETE')
