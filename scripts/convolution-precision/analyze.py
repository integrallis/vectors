"""Assess every fixed host and CPU stratum; no pooling historical or different-CPU rows."""
from pathlib import Path
import json, math, random, statistics, sys
P=json.loads(Path(__file__).with_name('protocol.json').read_text())
def interval(groups, seed):
    rng=random.Random(seed); result=[]
    for _ in range(P['bootstrap_samples']):
        means=[]
        for _ in groups:
            group=groups[rng.randrange(len(groups))]
            means.append(statistics.mean(group[rng.randrange(len(group))] for _ in group))
        result.append(math.exp(statistics.mean(means)))
    result.sort();return [result[int(.025*len(result))],result[int(.975*len(result))-1]]
def assess(hosts):
    results=[]
    for spec in P['strata']:
        cpu=spec['id']; expected_pairs=spec['pairs_per_host']
        selected=sorted([h for h in hosts if h['stratum']==cpu],key=lambda h:h['slot'])
        if len(selected)!=spec['hosts']:raise ValueError('Incomplete required stratum '+cpu)
        assert sorted(h['slot'] for h in selected)==list(range(spec['hosts'])),'Missing or duplicate host slot'
        logs=[]; cold=[];allocation=True
        for h in selected:
            assert h['cpu']==spec['cpu'],'CPU mismatch'
            assert all(h['actual'].get(k)==spec[k] for k in ['useAVX','useSVE','maxVectorSize'] if k in spec),'SIMD capability mismatch'
            rows=h['rows'];assert len(rows)==2*expected_pairs,'Incomplete host'
            keyed={(r['pair'],r['arm']):r for r in rows};assert len(keyed)==2*expected_pairs,'Duplicate pair arm'
            pairs=[];coldpairs=[]
            for pair in range(expected_pairs):
                a,b=(keyed[pair,k] for k in ['candidate','published'])
                for row in [a,b]:
                    blocks=row['blocks'];assert len(blocks)==5
                    assert all(v['nanos']>=1000000000 and v['calls']>0 and v['calls']%17==0 for v in blocks)
                    assert row['measured']==sum(v['calls'] for v in blocks)
                    assert row['channels']==P['channels'] and row['kernel']==P['kernel'] and row['warmup']==P['warmup_calls']
                    assert row['sha256']==P[row['arm']+'_sha256']
                    assert math.isfinite(row['hotNanosPerCall']) and row['hotNanosPerCall']>0
                assert a['bits']==b['bits'] and a['fma']==b['fma'],'Different numeric mode'
                assert not(a['coldOracleMismatches'] or a['hotOracleMismatches'] or a['coldHotMismatches'])
                pairs.append(math.log(a['hotNanosPerCall']/b['hotNanosPerCall']))
                coldpairs.append(math.log(a['coldBytes']/b['coldBytes']))
                allocation &= a['coldBytes']<=b['coldBytes'] and a['hotBytesPerCall']<=b['hotBytesPerCall']+P['warm_allocation_absolute_tolerance_bytes']
            logs.append(pairs);cold.append(coldpairs)
        ci=interval(logs,P['seed']);cc=interval(cold,P['seed']+1)
        gates={'warmLatency':ci[1]<P['latency_upper_limit'],'warmLatencyPrecision':ci[1]-ci[0]<=P['latency_interval_width_limit'],'coldAllocation':cc[1]<=P['cold_allocation_upper_limit'],'individualAllocation':allocation}
        results.append({'cpu':cpu,'hosts':len(selected),'pairs':expected_pairs*len(selected),'estimate':math.exp(statistics.mean(statistics.mean(g) for g in logs)),'ci95':ci,'coldAllocationCi95':cc,'gates':gates,'passed':all(gates.values())})
    return results
if __name__=='__main__':
    root,out=map(Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=True);hosts={}
    for f in sorted(root.rglob('manifest.json')):
        m=json.loads(f.read_text());assert m['protocol']==P,'Protocol mismatch';h=m['hardware'];key=(m['runId'],h['target'],h['attempt'])
        if key not in hosts:hosts[key]={'cpu':h['cpu'],'actual':h['actual'],'stratum':h['stratum'],'slot':h['slot'],'manifest':m,'rows':[],'seen':{}}
        else:assert hosts[key]['manifest']==m,'Host provenance changed'
        for rowfile in f.parent.glob('segment-*/*-candidate.json'):
            for q in [rowfile,rowfile.with_name(rowfile.name.replace('-candidate','-published'))]:
                row=json.loads(q.read_text());rk=(row['pair'],row['arm'])
                if rk in hosts[key]['seen']:assert hosts[key]['seen'][rk]==row,'Conflicting checkpoint duplicate'
                else:hosts[key]['seen'][rk]=row;hosts[key]['rows'].append(row)
    admissions={}
    for f in sorted(root.rglob('hardware-admission.json')):
        h=json.loads(f.read_text());key=(h['target'],h['attempt'])
        if key in admissions:assert admissions[key]==h,'Conflicting hardware identity'
        admissions[key]=h
    report={'candidateSha256':P['candidate_sha256'],'qualified':False,'scope':'Component CPU-stratified gate only; whole-model and QA remain separate.'}
    try:
        assert len({k[0] for k in hosts})==1,'Mixed run provenance'
        assert {h['runId'] for h in admissions.values()}=={k[0] for k in hosts},'Mixed admission run provenance'
        assert set(admissions)=={(arch,i) for arch,n in P['admission_attempts'].items() for i in range(n)},'Missing admission receipt'
        assert all(any(k[1:]==key for k in hosts) for key,h in admissions.items() if h['accepted']),'Admitted host missing manifest'
        report['results']=assess([hosts[k] for k in sorted(hosts)]);report['passed']=all(r['passed'] for r in report['results']);report['verdict']='COMPONENT_GATES_PASSED' if report['passed'] else 'NOT_PASSED'
    except (AssertionError,ValueError,KeyError) as e:report.update(verdict='INCOMPLETE_OR_INVALID',error=str(e),passed=False)
    (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));sys.exit(0 if report['passed'] else 1)
