"""Run one prospectively assigned checkpoint; never retry or change counts from results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from protocol import *

parser = argparse.ArgumentParser()
for name in ['candidate','old','published','output']: parser.add_argument(name,type=Path)
parser.add_argument('--segment',type=int,required=True)
args=parser.parse_args()
if not 0 <= args.segment < 5: raise ValueError('undeclared checkpoint')
args.output.mkdir(exist_ok=True,parents=True)
segment=args.output/f'segment-{args.segment}'
segment.mkdir(exist_ok=True)
if list(segment.glob('*-block*.json')): raise RuntimeError('Refusing repeated checkpoint')
commands=[]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(label, command, timeout=180):
    receipt = {'label': label, 'command': list(map(str, command)), 'startedUnix': time.time()}
    commands.append(receipt)
    try:
        with (segment / f'{label}.log').open('w') as log:
            process = subprocess.run(receipt['command'], stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        receipt['exitCode'] = process.returncode
        if process.returncode: raise RuntimeError(f'{label}: exit{process.returncode}')
    except Exception as error:
        receipt['error'] = repr(error)
        raise
    finally:
        receipt['elapsedSeconds'] = time.time() - receipt['startedUnix']
        (segment / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')


def core(root):
    jars = [p.resolve() for p in (root / 'vectors-core/build/libs').glob('vectors-core-*.jar') if not p.name.endswith(('-sources.jar', '-javadoc.jar'))]
    assert len(jars) == 1, jars
    return jars[0]


jars={arm:core(getattr(args,arm)) for arm in ARTIFACTS}
benchmark=Path('vectors-bench/src/jmh/java/com/integrallis/vectors/bench/BFloat16MatrixBenchmark.java')
assert all((getattr(args,a)/benchmark).read_bytes()==(args.published/benchmark).read_bytes() for a in ARTIFACTS)
manifest={'protocol':PROTOCOL,'cpuCount':os.cpu_count(),'jdk':subprocess.check_output(['java','-version'],stderr=subprocess.STDOUT).decode(),'artifacts':{},'benchmarkSha256':digest(args.published/benchmark)}
assert manifest['cpuCount']>=4 and '25.0.3' in manifest['jdk']
for arm,jar in jars.items():
 assert digest(jar)==ARTIFACTS[arm],(arm,digest(jar))
 manifest['artifacts'][arm]={'sha256':digest(jar),'source':subprocess.check_output(['git','-C',str(getattr(args,arm)),'rev-parse','HEAD']).decode().strip()}
path=args.output/'manifest.json'
if path.exists(): assert json.loads(path.read_text())==manifest
else: path.write_text(json.dumps(manifest,indent=2)+'\n')
libs = args.output / 'jmh-libs'
libs.mkdir(exist_ok=True)
dependencies = [
    ('org/openjdk/jmh', 'jmh-core', '1.37', 'dc0eaf2bbf0036a70b60798c785d6e03a9daf06b68b8edb0f1ba9eb3421baeb3'),
    ('org/openjdk/jmh', 'jmh-generator-annprocess', '1.37', '6a5604b5b804e0daca1145df1077609321687734a8b49387e49f10557c186c77'),
    ('net/sf/jopt-simple', 'jopt-simple', '5.0.4', 'df26cc58f235f477db07f753ba5a3ab243ebe5789d9f89ecf68dd62ea9a66c28'),
    ('org/apache/commons', 'commons-math3', '3.6.1', '1e56d7b058d28b65abd256b8458e3885b674c1d588fa43cd7d1cbb9c7ef2b308'),
]
for group, name, version, expected in dependencies:
    path = libs / f'{name}-{version}.jar'
    if not path.exists(): urllib.request.urlretrieve(f'https://repo.maven.apache.org/maven2/{group}/{name}/{version}/{path.name}', path)
    assert digest(path) == expected, path
classes = args.output / 'jmh-classes'
if args.segment == 0:
    classes.mkdir(exist_ok=True)
    run('compile-unchanged-benchmark', ['javac', '--add-modules', 'jdk.incubator.vector', '-cp',
        f'{jars["published"]}{os.pathsep}{libs}/*', '-processorpath', os.pathsep.join(str(p) for p in sorted(libs.glob('*.jar'))), '-d', classes, args.published / benchmark])
else:
    assert classes.exists(), 'First checkpoint did not compile benchmark'
for block in range(args.segment*3,args.segment*3+3):
 order=ARMS[block%5:]+ARMS[:block%5]
 for label in order:
  artifact='old' if label=='old-plain' else ('published' if label=='published-plain' else 'candidate')
  name=f'{label}-block{block:02d}'
  flags='-XX:ActiveProcessorCount=4 -Xmx3g'
  if label in ['candidate-log','candidate-assembly']:
   flags+=' -XX:+UnlockDiagnosticVMOptions -XX:+LogCompilation -XX:LogFile='+str((segment/(name+'-hotspot-%p.xml')).resolve())
  if label=='candidate-assembly': flags+=' -XX:+PrintCompilation -XX:CompileCommand=print,com.integrallis.vectors.core.BFloat16Matrix::*'
  print(f'START block={block} arm={label}',flush=True)
  run(name,['java','--add-modules','jdk.incubator.vector','-cp',f'{jars[artifact]}{os.pathsep}{classes}{os.pathsep}{libs}/*','org.openjdk.jmh.Main','BFloat16MatrixBenchmark.mappedBfloat16MatVec$','-f','1','-wi','5','-i','5','-w','1s','-r','1s','-p','shape=4864x896','-jvmArgsAppend',flags,'-prof','gc','-rf','json','-rff',segment/f'{name}.json'])
  records=json.loads((segment/f'{name}.json').read_text());assert len(records)==1
  record=records[0];assert record['params']['shape']=='4864x896' and record['primaryMetric']['scoreUnit']=='us/op' and math.isfinite(record['primaryMetric']['score'])
 print(f'BLOCK_COMPLETE {block}',flush=True)
(segment/'STATUS').write_text('COMPLETE\n')
