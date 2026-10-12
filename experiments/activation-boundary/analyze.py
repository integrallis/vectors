import json,math,pathlib,random,re,statistics
p=pathlib.Path(__file__).parent;d=p/'remote-v2';results={}
def summary(values):
 rng=random.Random(724);logs=[math.log(x) for x in values];boots=sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(10000));return {'candidateOverControl':math.exp(statistics.mean(logs)),'bootstrap95':[boots[250],boots[9750]],'pairs':len(values),'pairRatios':values}
metrics=['prefillNanos','decodeNanos','modelNanos','javaAllocatedBytes'];ratios={k:[] for k in metrics+['loadNanos','peakRssKb']};allocations={a:[] for a in ['native','boundary']}
for pair in range(10):
 rows={a:[json.loads(l) for l in (d/f'pair{pair}-{a}.stdout').read_text().splitlines() if l.startswith('{')] for a in allocations}
 req={a:[r for r in rows[a] if r.get('kind')=='request' and r['repeat']>=0] for a in rows};assert all(len(v)==3 for v in req.values())
 for a,rs in req.items():allocations[a].extend(r['javaAllocatedBytes'] for r in rs)
 for x,y in zip(req['native'],req['boundary']):
  assert x['id']==y['id'] and x['fullLogitSha256']==y['fullLogitSha256'] and x['greedyTokens']==y['greedyTokens']
 for k in metrics:ratios[k].append(math.exp(statistics.mean(math.log(y[k]/x[k]) for x,y in zip(req['native'],req['boundary']))))
 loads={a:next(r['nanos'] for r in rows[a] if r.get('kind')=='load') for a in rows}
 ratios['loadNanos'].append(loads['boundary']/loads['native'])
 rss={a:int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',(d/f'pair{pair}-{a}.stderr').read_text()).group(1)) for a in rows};ratios['peakRssKb'].append(rss['boundary']/rss['native'])
results={'phase':'activation-boundary-v2','ratio':'boundary/control; lower is better','resampling':'10000 paired-process cluster draws seed724','correctness':json.loads((d/'correctness.json').read_text()),'metrics':{k:summary(v) for k,v in ratios.items()},'allocatedBytes':{k:{'minimum':min(v),'maximum':max(v),'mean':statistics.mean(v)} for k,v in allocations.items()}}
(p/'paired-summary.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
