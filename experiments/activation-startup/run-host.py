import hashlib,json,os,pathlib,subprocess,time
out=pathlib.Path('/work/out');config=json.loads(pathlib.Path('confirmation.json').read_text());model=config['models'][0];commands=[]
def status(value):
 (out/'STATUS').write_text(value+'\n');subprocess.run(['aws','s3','cp',str(out),os.environ['RESULT_URI'],'--recursive','--only-show-errors'],check=True)
def run(cmd,name,parse=True):
 start=time.monotonic()
 with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:r=subprocess.run(cmd,stdout=so,stderr=se,timeout=300)
 commands.append({'name':name,'command':cmd,'exitCode':r.returncode,'wallSeconds':time.monotonic()-start});(out/'commands.json').write_text(json.dumps(commands,indent=2));assert r.returncode==0,name
 return [json.loads(l) for l in (out/(name+'.stdout')).read_text().splitlines() if l.startswith('{')] if parse else []
for arm in ['native','published']:
 status('BUILDING_'+arm);run(['/root/.cargo/bin/cargo','build','--locked','--release','--manifest-path',f'rust/{arm}/Cargo.toml'],'rust-'+arm,False)
run(['curl','-fL','--retry','3',model['downloadUri'],'-o','model.gguf'],'download',False);assert hashlib.file_digest(open('model.gguf','rb'),'sha256').hexdigest()==model['sha256']
base=['/opt/jdk/bin/java','--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xms2g','-Xmx10g']
for pair in range(100):
 arms=['stable','native','published'];order=arms[pair%3:]+arms[:pair%3]
 if pair%2:order.reverse()
 for arm in order:
  native='published' if arm=='published' else 'native'
  rows=run(['/usr/bin/time','-v',*base,'-cp',f'lib/{arm}/*:route-proof.jar','com.integrallis.models.backend.purejava.qwen35.ProductionRouteProof','load','model.gguf',f'/work/rust/{native}/target/release/libjmodels_kernels.so','data/tasks.json'],f'pair{pair}-{arm}')
  assert len([r for r in rows if r.get('kind')=='load'])==1
 if pair%10==9:status('PAIRS_'+str(pair+1))
status('COMPLETE')
