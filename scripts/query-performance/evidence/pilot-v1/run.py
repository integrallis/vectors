#!/usr/bin/env python3
"""Alternating fresh JVM query comparisons. No build or other benchmark runs concurrently."""
import argparse, hashlib, json, os, statistics, subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path)
p.add_argument('archive',type=Path);p.add_argument('output',type=Path)
p.add_argument('--rounds',type=int,default=3)
p.add_argument('--baseline-sha',required=True);p.add_argument('--candidate-sha',required=True)
p.add_argument('--modes',default='flat,heap,mapped')
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
java=Path(os.environ['JAVA_HOME'])/'bin/java'; source=Path(__file__).with_name('QueryPerformance.java').resolve()
cp={k:os.pathsep.join(str(j.resolve()) for j in sorted(d.glob('*.jar'))) for k,d in [('baseline',a.baseline),('candidate',a.candidate)]}
classes=out/'classes';classes.mkdir()
subprocess.run([str(java.with_name('javac')),'--add-modules=jdk.incubator.vector','-cp',cp['baseline'],'-d',str(classes),str(source)],check=True)
r=a.archive.resolve()
real=r/'final-replay/candidate-0-collection/gen-0000000000000004'
cases=[('glove',r/'inputs/glove-100-angular-train.fbin',r/'inputs/glove-100-angular-test.fbin',r/'combined-results/glove-100-angular-t1-s17-baseline.graph.bin','COSINE'),('dbpedia',real,real,real/'graph.bin','COSINE'),('fashion',r/'inputs/fashion-mnist-784-euclidean-train.fbin',r/'inputs/fashion-mnist-784-euclidean-test.fbin',r/'combined-results/fashion-mnist-784-euclidean-t1-s17-baseline.graph.bin','EUCLIDEAN')]
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
files=set([java,source])
for _,data,query,graph,_ in cases:
 for f in [data,query,graph]:
  files.update([f/'manifest.bin',f/'vectors.bin'] if f.is_dir() else [f])
for directory in [a.baseline,a.candidate]:files.update(directory.glob('*.jar'))
prov={'baseline_sha':a.baseline_sha,'candidate_sha':a.candidate_sha,'files':{str(f):sha(f) for f in sorted(files)}}
(out/'provenance.json').write_text(json.dumps(prov,indent=2))
commands=[];samples={};digests={}
for rep in range(a.rounds):
 for name,data,queries,graph,metric in cases:
  for mode in a.modes.split(','):
   for arm in (['baseline','candidate'] if rep%2==0 else ['candidate','baseline']):
    label=f'{name}-{mode}-{arm}-{rep}';print(label,flush=True)
    cmd=[str(java),'--add-modules=jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xms3g','-Xmx3g','-cp',str(classes)+os.pathsep+cp[arm],'QueryPerformance',name,str(data),str(queries),str(graph),metric,mode]
    commands.append(cmd);(out/'commands.json').write_text(json.dumps(commands,indent=2))
    with (out/(label+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    values={}
    for line in (out/(label+'.log')).read_text().splitlines():
     if not line.startswith('RESULT,'):continue
     _,key,sample,ns,h,nq=line.split(',')
     values.setdefault(key,[]).append(int(ns)/int(nq))
     digests.setdefault(arm,{}).setdefault(key,set()).add(h)
    assert values,label
    for key,ns in values.items():
     assert len(ns)==7,(key,len(ns))
     samples.setdefault(key,{}).setdefault(arm,[]).append(statistics.median(ns))
    summary={}
    for key,arms in samples.items():
     if len(arms)!=2:continue
     b=statistics.median(arms['baseline']);c=statistics.median(arms['candidate'])
     parity=digests['baseline'][key]==digests['candidate'][key] and len(digests['baseline'][key])==1
     summary[key]={'baseline_us':b/1e3,'candidate_us':c/1e3,'less_time_percent':100*(1-c/b),'exact_parity':parity,'forks_ns_per_query':arms}
     assert parity,key
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
assert all(sha(Path(f))==h for f,h in prov['files'].items()),'input/runtime changed'
