#!/usr/bin/env python3
"""Check real Maven consumers of staged optional runtimes, independently of Gradle metadata."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repository', type=Path, required=True)
parser.add_argument('--output', type=Path, default=Path('build/maven-runtime-smoke'))
args = parser.parse_args()
repository = args.repository.resolve()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
(out / 'results.json').unlink(missing_ok=True)
root = Path(__file__).resolve().parent.parent
version = re.search(r'^version\s*=\s*(\S+)', (root / 'gradle.properties').read_text(), re.M).group(1)
netty = {f'io.netty:{name}': '4.1.138.Final' for name in (
    'netty-buffer', 'netty-codec', 'netty-codec-http', 'netty-codec-http2',
    'netty-common', 'netty-handler', 'netty-resolver', 'netty-transport',
    'netty-transport-classes-epoll', 'netty-transport-native-unix-common')}
jackson = {
    'com.fasterxml.jackson.core:jackson-core': '2.21.7',
    'com.fasterxml.jackson.core:jackson-annotations': '2.21',
    'com.fasterxml.jackson.core:jackson-databind': '2.21.7',
}
cases = {
    'vectors-storage-s3': netty,
    'vectors-db-arrow': {**jackson, 'com.fasterxml.jackson.datatype:jackson-datatype-jsr310': '2.21.7'},
    'vectors-vcr-serde-jackson': jackson,
}
results = {}
# Each execution gets a fresh Maven cache so an earlier candidate with the same release GAV
# cannot conceal a broken POM. Each module is consumed alone to prevent one runtime from
# accidentally repairing another runtime's dependency graph through nearest-wins mediation.
with tempfile.TemporaryDirectory(prefix='vectors-maven-consumer-') as temporary:
    for artifact, expected in cases.items():
        staged = repository / 'com/integrallis' / artifact / version / f'{artifact}-{version}.pom'
        if not staged.is_file():
            raise FileNotFoundError(staged)
        project = ET.Element('project', xmlns='http://maven.apache.org/POM/4.0.0')
        for key, value in [('modelVersion', '4.0.0'), ('groupId', 'audit'),
                           ('artifactId', artifact + '-consumer'), ('version', '1')]:
            ET.SubElement(project, key).text = value
        repo = ET.SubElement(ET.SubElement(project, 'repositories'), 'repository')
        ET.SubElement(repo, 'id').text = 'vectors-candidate'
        ET.SubElement(repo, 'url').text = repository.as_uri()
        dependency = ET.SubElement(ET.SubElement(project, 'dependencies'), 'dependency')
        for key, value in [('groupId', 'com.integrallis'), ('artifactId', artifact), ('version', version)]:
            ET.SubElement(dependency, key).text = value
        case = out / artifact
        case.mkdir(exist_ok=True)
        pom = case / 'pom.xml'
        ET.ElementTree(project).write(pom, encoding='utf-8', xml_declaration=True)
        tree = case / 'dependencies.txt'
        # Remove a previous result before execution; a failed Maven command must never reuse it.
        tree.unlink(missing_ok=True)
        command = ['mvn', '-B', '-f', str(pom), f'-Dmaven.repo.local={temporary}/repository',
                   'org.apache.maven.plugins:maven-dependency-plugin:3.8.1:tree',
                   f'-DoutputFile={tree}']
        with (case / 'maven.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        resolved = dict(re.findall(r'([\w.\-]+:[\w.\-]+):jar:([^:\s]+):', tree.read_text()))
        mismatches = {key: {'expected': value, 'actual': resolved.get(key)}
                      for key, value in expected.items() if resolved.get(key) != value}
        results[artifact] = {'expected': expected, 'mismatches': mismatches}
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        if mismatches:
            raise RuntimeError(f'{artifact}: unexpected Maven dependency versions: {mismatches}')
        print(f'{artifact}: {len(expected)} patched runtime dependencies verified', flush=True)
