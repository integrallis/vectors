#!/usr/bin/env python3
"""Recompute summaries from raw logs; fail on missing/duplicate samples or result mismatch."""
import argparse, json, math, re, statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--rounds',type=int,default=3)
a=p.parse_args();samples={};digests={};counts={}
for log in sorted(a.output.glob('*.log')):
 match=re.search(r'-(baseline|candidate)-(\d+)\.log$',log.name)
 if not match:continue
 arm,rep=match.group(1),int(match.group(2));groups={}
 for line in log.read_text().splitlines():
  if not line.startswith('RESULT,'):continue
  _,case,index,ns,digest,nq=line.split(',');index=int(index);ns=int(ns);nq=int(nq)
  assert ns>0 and nq>0,line
  values=groups.setdefault(case,{})
  assert index not in values,(log,index)
  values[index]=ns/nq
  digests.setdefault(case,set()).add(digest)
  counts.setdefault(case,set()).add(nq)
 assert groups,log
 for case,values in groups.items():
  assert set(values)==set(range(7)),(log,case)
  armvalues=samples.setdefault(case,{}).setdefault(arm,{})
  assert rep not in armvalues,(log,case)
  armvalues[rep]=statistics.median(values.values())
reported=json.loads((a.output/'summary.json').read_text());assert set(reported)==set(samples)
for case,arms in samples.items():
 assert set(arms)=={'baseline','candidate'},case
 assert len(digests[case])==len(counts[case])==1,case
 for arm,reps in arms.items():assert set(reps)==set(range(a.rounds)),(case,arm)
 b=statistics.median(arms['baseline'].values());c=statistics.median(arms['candidate'].values())
 expected={'baseline_us':b/1e3,'candidate_us':c/1e3,'less_time_percent':100*(1-c/b)}
 for key,value in expected.items():assert math.isclose(reported[case][key],value,rel_tol=1e-12,abs_tol=1e-12),(case,key)
 assert reported[case]['exact_parity'] is True
 for arm in arms:assert reported[case]['forks_ns_per_query'][arm]==[arms[arm][i] for i in range(a.rounds)]
 print(f"{case}: {expected['less_time_percent']:.3f}% less time; identical result digest")
print(f'Audited {len(samples)} cases across {a.rounds} pairs.')
