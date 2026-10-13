"""Assess only the prospectively declared CPU/case cells; retain every fork."""
from pathlib import Path
import json,re,sys,math,statistics
from estimator import hierarchical_interval,latency_verdict
P=json.loads(Path(__file__).with_name('protocol.json').read_text())
def assess(hosts):
 results=[]
 expected={(s['id'],i) for s in P['strata'] for i in range(s['hosts'])};assert set(hosts)==expected,'Missing or unexpected host'
 for spec in P['strata']:
  keys=[(spec['id'],i) for i in range(spec['hosts'])]
  for key in keys:
   h=hosts[key];m=h['manifest'];hw=m['hardware'];assert hw['stratum']==key[0] and hw['slot']==key[1] and hw['cpu']==spec['cpu']
   assert all(hw['actual'][k]==spec[k] for k in ['useAVX','maxVectorSize'])
   expected_rows={(case,i,arm) for case,n in spec['counts'].items() for i in range(n) for arm in ['baseline','candidate']};assert set(h['rows'])==expected_rows,'Incomplete or unexpected case/pair'
   for (case,i,arm),r in h['rows'].items():
    method,shape=P['cases'][case];assert r['benchmark'].endswith('.'+method) and r['params']['shape']==shape
    assert r['primaryMetric']['scoreUnit']=='us/op' and math.isfinite(r['primaryMetric']['score']) and r['primaryMetric']['score']>0
    assert r['warmupIterations']==5 and r['measurementIterations']==5 and r['forks']==1 and r['warmupTime']=='1 s' and r['measurementTime']=='1 s'
    assert r['jdkVersion']=='25.0.3' and '-XX:ActiveProcessorCount=4' in r['jvmArgs'] and '-Xmx3g' in r['jvmArgs']
  for case,count in spec['counts'].items():
   logs={};deltas={};alloc={'baseline':[],'candidate':[]}
   for key in keys:
    logs[str(key)]=[];deltas[str(key)]=[]
    for i in range(count):
     a,b=(hosts[key]['rows'][case,i,arm] for arm in ['candidate','baseline']);logs[str(key)].append(math.log(a['primaryMetric']['score']/b['primaryMetric']['score']))
     aa,bb=(v['secondaryMetrics']['gc.alloc.rate.norm']['score'] for v in [a,b]);deltas[str(key)].append(aa-bb);alloc['candidate'].append(aa);alloc['baseline'].append(bb)
   t=latency_verdict(hierarchical_interval(logs,resamples=P['bootstrap_samples'],seed=P['seed']));results.append({'stratum':spec['id'],'case':case,'hosts':len(keys),'pairs':count*len(keys),'latency':t,'allocation':{'baseline':statistics.mean(alloc['baseline']),'candidate':statistics.mean(alloc['candidate']),'pairedDifference':hierarchical_interval(deltas,resamples=P['bootstrap_samples'],seed=P['seed'],exponentiate=False)}})
 return results
if __name__=='__main__':
 root,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=True,parents=True);hosts={};admissions={};report={'qualified':False,'scope':P['scope'],'protocol':P}
 try:
  for f in sorted(root.rglob('manifest.json')):
   m=json.loads(f.read_text());assert m['protocol']==P;hw=m['hardware'];key=(hw['stratum'],hw['slot']);h=hosts.setdefault(key,{'manifest':m,'rows':{}});assert h['manifest']==m,'Conflicting host identity'
   for q in sorted(f.parent.glob('segment-*/*-pair*.json')):
    if q.name.endswith('.command.json'):continue
    match=re.fullmatch(r'([a-z_]+)-pair(\d+)-(baseline|candidate)\.json',q.name);assert match
    rk=(match[1],int(match[2]),match[3]);r=json.loads(q.read_text());assert len(r)==1
    if rk in h['rows']:assert h['rows'][rk]==r[0],'Conflicting checkpoint copy'
    else:h['rows'][rk]=r[0]
  for f in sorted(root.rglob('hardware-admission.json')):
   h=json.loads(f.read_text());key=(h['target'],h['attempt'])
   if key in admissions:assert admissions[key]==h
   admissions[key]=h
  runids={h['manifest']['runId'] for h in hosts.values()};assert len(runids)==1 and {a['runId'] for a in admissions.values()}==runids,'Mixed run provenance'
  assert set(admissions)=={(a,i) for a,n in P['admission_attempts'].items() for i in range(n)},'Missing admission receipt'
  for h in admissions.values():
   if h['accepted']:assert (h['stratum'],h['slot']) in hosts and hosts[h['stratum'],h['slot']]['manifest']['hardware']==h,'Missing admitted host'
  report['results']=assess(hosts);report['passed']=all(r['latency']['direction']=='faster' and r['latency']['precisionTargetMet'] for r in report['results']);report['verdict']='TARGETED_CASES_PASS' if report['passed'] else 'NOT_PASSED'
 except (AssertionError,ValueError,KeyError) as e:report.update(passed=False,verdict='INCOMPLETE_OR_INVALID',error=str(e))
 (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));sys.exit(0 if report['passed'] else 1)
