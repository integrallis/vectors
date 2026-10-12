"""Bootstrap whole process blocks; never treat requests within one JVM as independent."""
import json, math, pathlib, random, re, statistics
root=pathlib.Path(__file__).parent
raw=root/'remote'
records=[]
for pair in range(100):
 for arm in ('stable','native','published'):
  name=f'pair{pair}-{arm}'
  stdout=raw/(name+'.stdout');stderr=raw/(name+'.stderr')
  row=next(json.loads(s) for s in stdout.read_text().splitlines() if s.startswith('{') and json.loads(s).get('kind')=='load')
  rss=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',stderr.read_text())[1])
  records.append(dict(pair=pair,arm=arm,modelLoadNanos=row['nanos'],loadOnlyPeakRssKb=rss))
(root/'loads.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
lookup={(r['pair'],r['arm']):r for r in records};summary=[]
for baseline in ('stable','published'):
 values=[math.log(lookup[(p,'native')]['modelLoadNanos']/lookup[(p,baseline)]['modelLoadNanos']) for p in range(100)]
 rng=random.Random(724);samples=sorted(math.exp(sum(rng.choices(values,k=100))/100) for _ in range(10000))
 summary.append(dict(baseline=baseline,candidate='final',pairs=100,metric='modelLoadNanos',ratio=math.exp(statistics.mean(values)),ci95=[samples[250],samples[9750]],resampling='100 whole process blocks; seed724; 10000 draws'))
(root/'startup-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
