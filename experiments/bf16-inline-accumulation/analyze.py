from pathlib import Path
import json,statistics,sys
base=Path('/Users/briansam-bodden/Code/java-ai/projects/models-gdn-proof/benchmark-results/2026-10-11-gdn-proof/remote-bf16-inline-v1')/(sys.argv[1] if len(sys.argv)>1 else 'x86');result={'scope':'two-pair diagnostic, not full release qualification','pairs':[]}
for pair in [0,1]:
 arms={}
 for arm in ['inline','control']:
  f=base/f'bonsai2-pair{pair}-{arm}.stdout'
  if not f.exists():continue
  records=[]
  for line in f.read_text().splitlines():
   try:j=json.loads(line)
   except ValueError:continue
   if j.get('kind')=='request':records.append(j)
  arms[arm]=records
 row={'pair':pair,'completedRequests':{a:len(r) for a,r in arms.items()}}
 for arm,records in arms.items():
  warm=[r for r in records if r['repeat']>=0]
  if warm:row[arm]={k:statistics.mean(r[k] for r in warm) for k in ['javaAllocatedBytes','prefillNanos','decodeNanos','requestNanos']}
 if all(a in arms for a in ['inline','control']):
  a,b=arms['inline'],arms['control'];row['fullSnapshotMatches']=[x['fullLogitSha256']==y['fullLogitSha256'] and x['greedyTokens']==y['greedyTokens'] and x['id']==y['id'] and x['repeat']==y['repeat'] for x,y in zip(a,b)]
  if len(a)==4 and len(b)==4 and all(a in row for a in ['inline','control']):row['inlineControlRatios']={k:row['inline'][k]/row['control'][k] for k in row['inline']}
 result['pairs'].append(row)
(base/'diagnostic-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
