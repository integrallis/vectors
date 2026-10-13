#!/usr/bin/env python3
"""Format-check, clean-build, and record an immutable runtime before freezing a protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    java_home = Path(os.environ['JAVA_HOME'])
    source_root = root / 'vectors-core/src/main'
    sources = sorted(p for p in source_root.rglob('*') if p.is_file())
    paths = [str(p.relative_to(root)) for p in sources]
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *paths], cwd=root, check=True)
    command = ['./gradlew', ':vectors-core:spotlessJavaCheck', ':vectors-core:clean', ':vectors-core:jar',
               '--max-workers=2', '--no-daemon', '-Porg.gradle.java.installations.auto-detect=false',
               '-Porg.gradle.java.installations.auto-download=false',
               '-Porg.gradle.java.installations.paths=' + str(java_home)]
    with (output / 'clean-build.log').open('w') as stream:
        subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
    jars = [p for p in (root / 'vectors-core/build/libs').glob('*.jar')
            if not p.name.endswith(('-sources.jar', '-javadoc.jar'))]
    if len(jars) != 1:
        raise ValueError('Expected exactly one clean-built runtime JAR')
    jar = jars[0]
    classes = {}
    with zipfile.ZipFile(jar) as archive:
        for name in ['PanamaVectorUtilSupport', 'ReproducibleConvolutionSilu']:
            qualified = 'com.integrallis.vectors.core.' + name
            code = subprocess.check_output([str(java_home / 'bin/javap'), '-classpath', str(jar),
                                            '-p', '-c', qualified])
            (output / (name + '.javap.txt')).write_bytes(code)
            classes[qualified] = {'classFileSha256': digest(archive.read(qualified.replace('.', '/') + '.class')),
                                  'javapCodeSha256': digest(code)}
    manifest = {
        'sourceRevision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'sourceFileSha256': {str(p.relative_to(root)): digest(p.read_bytes()) for p in sources},
        'jarSha256': digest(jar.read_bytes()), 'classes': classes, 'buildCommand': command,
        'java': subprocess.check_output([str(java_home / 'bin/java'), '-version'], stderr=subprocess.STDOUT, text=True),
    }
    shutil.copy2(jar, output / jar.name)
    (output / 'freeze.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'sourceRevision': manifest['sourceRevision'], 'jarSha256': manifest['jarSha256']}))


if __name__ == '__main__':
    main()
