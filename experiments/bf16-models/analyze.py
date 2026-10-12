"""Process-paired artifact comparison; incomplete or imprecise data cannot select a default."""
import json,math,pathlib,random,re,statistics,sys

folder=pathlib.Path(sys.argv[1]);output=pathlib.Path(sys.argv[2])
arms=['control','candidate','published']
metrics=['loadNanos','coldRequestNanos','prefillNanos','decodeNanos','requestNanos','peakRssKiB','javaAllocatedBytes']
results=[]
host_manifest=json.loads((folder/'pair-hosts.json').read_text()) if (folder/'pair-hosts.json').exists() else {}
def read(model,pair,arm):
    prefix=folder/f'{model}-pair{pair}-{arm}'
    execution=json.loads(pathlib.Path(str(prefix)+'.execution.json').read_text())
    assert execution['exitCode']==0,prefix
    rows=[json.loads(line) for line in pathlib.Path(str(prefix)+'.stdout').read_text().splitlines() if line.startswith('{')]
    loads=[r for r in rows if r['kind']=='load'];assert len(loads)==1
    requests=[r for r in rows if r['kind']=='request'];assert len(requests)==7
    warm=[r for r in requests if r['repeat']>=0];assert len(warm)==3
    rss=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',pathlib.Path(str(prefix)+'.stderr').read_text())
    assert rss,prefix
    values={'loadNanos':loads[0]['nanos'],'peakRssKiB':int(rss.group(1)),
        'coldRequestNanos':next(r['requestNanos'] for r in requests if r['repeat']==-4)}
    values.update({k:statistics.mean(r[k] for r in warm) for k in ['prefillNanos','decodeNanos','requestNanos','javaAllocatedBytes']})
    return values,requests

def paired(candidate,reference,hosts):
    logs=[math.log(c/r) for c,r in zip(candidate,reference)]
    rng=random.Random(716309)
    groups={host:[logs[n] for n,value in enumerate(hosts) if value==host] for host in sorted(set(hosts))}
    samples=sorted(math.exp(statistics.mean([value for values in groups.values() for value in rng.choices(values,k=len(values))])) for _ in range(20000))
    low,high=samples[500],samples[19499]
    return {'ratio':math.exp(statistics.mean(logs)),'ci95':[low,high],
        'precisionTargetMet':(high-low)<=0.02,
        'direction':'slower' if low>1 else 'faster' if high<1 else 'unresolved',
        'processRatios':[math.exp(x) for x in logs],
        'hostGroups':{host:{'pairs':len(values),'ratio':math.exp(statistics.mean(values))} for host,values in groups.items()}}

for model in (sys.argv[3:] or ['qwen-bf16','bonsai2']):
    hosts=[host_manifest.get(model,{}).get(str(pair),'single-host') for pair in range(10)]
    samples={arm:[] for arm in arms};changes=[];quality={arm:[] for arm in arms}
    for pair in range(10):
        records={}
        for arm in arms:
            values,records[arm]=read(model,pair,arm);samples[arm].append(values)
            quality[arm].extend(r for r in records[arm] if r['repeat']>=0)
        for arm in arms[1:]:
            for baseline,candidate in zip(records['control'],records[arm]):
                assert (baseline['id'],baseline['repeat'])==(candidate['id'],candidate['repeat'])
                changed=[k for k in ['inputTokens','fullLogitSha256','greedyTokens','prediction'] if baseline[k]!=candidate[k]]
                if changed:changes.append({'pair':pair,'arm':arm,'id':candidate['id'],'repeat':candidate['repeat'],'fields':changed,'controlPrediction':baseline['prediction'],'otherPrediction':candidate['prediction'],'expected':baseline['expected'],'controlGreedyTokens':baseline['greedyTokens'],'otherGreedyTokens':candidate['greedyTokens']})
    comparisons=[]
    for candidate,reference in [('candidate','control'),('candidate','published'),('control','published')]:
        comparisons.append({'candidate':candidate,'reference':reference,'metrics':{
            metric:paired([s[metric] for s in samples[candidate]],[s[metric] for s in samples[reference]],hosts) for metric in metrics}})
    quality_summary={}
    for arm,requests in quality.items():
        assert len(requests)==30 and len({r['id'] for r in requests})==30
        datasets=sorted({r['dataset'] for r in requests})
        quality_summary[arm]={dataset:{'count':sum(r['dataset']==dataset for r in requests),'correct':sum(r['dataset']==dataset and r['prediction']==r['expected'] for r in requests)} for dataset in datasets}
    results.append({'model':model,'processPairs':10,'warmRequestsPerArm':30,'quality':quality_summary,'changes':changes,'comparisons':comparisons})
output.write_text(json.dumps({'method':'geometric mean of process-paired ratios; host-stratified paired percentile bootstrap, 20000 resamples, fixed seed; each host retains its observed sample count; warm metrics average three requests per process',
    'results':results},indent=2)+'\n')
assert not any(c['arm']!='published' for r in results for c in r['changes']), 'candidate numeric contract mismatch'
print(output)
