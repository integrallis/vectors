"""Bounded hardware-only admission; atomic slots cannot be chosen using measurements."""
import json,os,re,subprocess,sys,urllib.request,urllib.error
from pathlib import Path
P=json.loads(Path(__file__).with_name('protocol.json').read_text())
arch,attempt=sys.argv[1],int(sys.argv[2]);assert arch in P['admission_attempts'] and 0<=attempt<P['admission_attempts'][arch]
rows=json.loads(subprocess.check_output(['lscpu','-J']))['lscpu'];cpu=next(r['data'] for r in rows if r['field'].rstrip(':')=='Model name')
flags=subprocess.check_output(['java','-XX:+UnlockDiagnosticVMOptions','-XX:+PrintFlagsFinal','-version'],stderr=subprocess.STDOUT,text=True)
def flag(name):
 m=re.search(r'\b'+name+r'\s+=\s+(\d+)',flags);return int(m[1]) if m else None
actual={'useAVX':flag('UseAVX'),'useSVE':flag('UseSVE'),'maxVectorSize':flag('MaxVectorSize')}
spec=next((s for s in P['strata'] if s['cpu']==cpu and all(actual[k]==s[k] for k in actual if k in s)),None)
h={'cpu':cpu,'target':arch,'attempt':attempt,'accepted':False,'actual':actual,'lscpu':rows,'reason':'unmatched hardware','runId':os.environ['GITHUB_RUN_ID']}
out=Path('evidence');out.mkdir(exist_ok=True)
try:
 if spec:
  h['stratum']=spec['id'];h['reason']='all declared host slots already occupied'
  leases=json.loads(os.environ['CONVOLUTION_ADMISSION_LEASES']);assert leases['cohort']==P['cohort']
  for slot,url in enumerate(leases['urls'][spec['id']]):
   body=json.dumps({'cohort':P['cohort'],'runId':h['runId'],'attempt':attempt,'arch':arch,'cpu':cpu,'actual':actual,'slot':slot}).encode()
   try:
    req=urllib.request.Request(url,data=body,headers={'If-None-Match':'*'},method='PUT')
    with urllib.request.urlopen(req,timeout=30) as response:assert response.status==200
    h.update(accepted=True,slot=slot,reason='atomic hardware-only admission');break
   except urllib.error.HTTPError as error:
    if error.code!=412:raise RuntimeError('Admission request failed with HTTP '+str(error.code)) from None
finally:
 (out/'hardware-admission.json').write_text(json.dumps(h,indent=2)+'\n')
 with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('accepted='+str(h['accepted']).lower()+'\n')
 print(json.dumps({k:v for k,v in h.items() if k!='lscpu'}))
