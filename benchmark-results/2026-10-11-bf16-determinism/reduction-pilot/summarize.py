#!/usr/bin/env python3
import argparse,json,pathlib,statistics
p=argparse.ArgumentParser();p.add_argument('directory',type=pathlib.Path);a=p.parse_args()
order=json.loads((a.directory/'order.json').read_text());groups={}
for i,arm in enumerate(order):
 records=json.loads((a.directory/f'{i}-{arm}.json').read_text())
 assert len(records)==4
 for r in records:
  key=(r['benchmark'].rsplit('.',1)[-1],r['params']['shape'])
  groups.setdefault(key,{}).setdefault(arm,[]).append(r)
results=[]
for (kernel,shape),arms in groups.items():
 base=statistics.mean(r['primaryMetric']['score'] for r in arms['baseline'])
 row={'kernel':kernel,'shape':shape,'arms':{}}
 for arm,records in arms.items():
  assert len(records)==2
  scores=[r['primaryMetric']['score'] for r in records]
  row['arms'][arm]={'microseconds':statistics.mean(scores),'latencyRatio':statistics.mean(scores)/base,'forkMicroseconds':scores,'bytesPerCall':statistics.mean(r['secondaryMetrics']['gc.alloc.rate.norm']['score'] for r in records)}
 results.append(row)
 print(kernel,shape,{k:round(v['latencyRatio'],4) for k,v in row['arms'].items()})
(a.directory/'summary.json').write_text(json.dumps({'results':results,'scope':'Diagnostic two-shape pilot; remaining shape and real-model checks required'},indent=2)+'\n')
