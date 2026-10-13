#!/usr/bin/env python3
"""Record actual generated code; these instrumented runs are not timing evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convolution-screen"))
from run import PROTOCOL, HERE, sha, execute

out = Path('diagnostic').resolve()
out.mkdir(exist_ok=True)
classes = out / 'classes'
classes.mkdir(exist_ok=True)
slf4j = next((Path.home() / '.gradle/caches/modules-2/files-2.1/org.slf4j/slf4j-api/2.0.17').glob('*/*.jar'))
manifest = {'protocol': PROTOCOL, 'runs': [], 'scope': 'Instrumented compiler diagnosis only; not performance qualification.'}
execute(['java', '-XX:+UnlockDiagnosticVMOptions', '-XX:+PrintFlagsFinal', '-version'], out / 'jvm-flags.log')
execute(['lscpu'], out / 'cpu.log')
for arm in PROTOCOL['arms']:
    jars = [p for p in Path(arm + '/vectors-core/build/libs').glob('*.jar')
            if not p.name.endswith(('-sources.jar', '-javadoc.jar'))]
    if len(jars) != 1 or sha(jars[0]) != PROTOCOL[arm + '_sha256']:
        raise ValueError(f'{arm}: actual artifact mismatch')
    jar = jars[0].resolve()
    cp = os.pathsep.join(map(str, [classes, jar, slf4j]))
    execute(['javac', '--add-modules', 'jdk.incubator.vector', '-cp', cp, '-d', str(classes),
             str(HERE / 'ConvolutionBoundaryProbe.java')], out / f'{arm}-compile.log')
    command = ['java', '--add-modules', 'jdk.incubator.vector', '-XX:ActiveProcessorCount=4',
               '-Xms256m', '-Xmx512m', '-XX:+UnlockDiagnosticVMOptions', '-XX:+PrintCompilation',
               '-XX:CompileCommand=print,com.integrallis.vectors.core.PanamaVectorUtilSupport::causalDepthwiseConv1dSilu']
    if arm == 'candidate':
        command += ['-XX:CompileCommand=print,com.integrallis.vectors.core.ReproducibleConvolutionSilu::apply',
                    '-XX:CompileCommand=print,com.integrallis.vectors.core.ReproducibleConvolutionSilu::applyScaled']
    command += ['-cp', cp, 'com.integrallis.vectors.core.ConvolutionBoundaryProbe']
    execute(command, out / f'{arm}-assembly.log', timeout=180)
    manifest['runs'].append({'arm': arm, 'sha256': sha(jar), 'command': command,
                            'revision': subprocess.check_output(['git', '-C', arm, 'rev-parse', 'HEAD'], text=True).strip()})
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
