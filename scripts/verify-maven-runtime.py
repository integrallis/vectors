#!/usr/bin/env python3
"""Verify every published runtime with independent Maven and Gradle consumers.

The input receipt defines the publication allowlist and expected versions. Consumers do not
import our build policy, add BOMs, use source substitution, or share a previous candidate cache.
Use --repository for staging, or --central after publication.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument('--repository', type=Path)
source.add_argument('--central', action='store_true')
parser.add_argument('--output', type=Path, default=Path('build/published-consumers'))
parser.add_argument('--receipt', type=Path, default=Path('build/reports/release-dependencies/train.json'))
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
(out / 'results.json').unlink(missing_ok=True)
receipt = json.loads(args.receipt.read_text())
if not receipt['modules'] or any(m['failures'] for m in receipt['modules']):
    raise ValueError('A successful dependency receipt is required')
policy = receipt['externalPolicy']
version = receipt['versions'][receipt['family']]
publications = {m['publication']['artifact']: m['publication'] for m in receipt['modules']}
modules = list(publications)
repository = args.repository.resolve().as_uri() if args.repository else 'https://repo.maven.apache.org/maven2'


def expected(group, artifact):
    if group == 'com.integrallis':
        if artifact.startswith('vectors'):
            return receipt['versions'].get('vectors', 'FORBIDDEN')
        if artifact == 'models' or artifact.startswith(('models-', 'backend-')):
            return receipt['versions'].get('models', 'FORBIDDEN')
    if group in ('org.modeljars', 'org.modeljars.composite'):
        return receipt['versions'].get('modeljars', 'FORBIDDEN')
    if group == 'org.slf4j':
        return policy['slf4j']
    if group.startswith('com.fasterxml.jackson'):
        return policy['jacksonAnnotations'] if artifact == 'jackson-annotations' else policy['jacksonBom']
    return policy.get('modules', {}).get(group + ':' + artifact) or policy.get('families', {}).get(group)


def validate(nodes, label):
    failures = []
    for node in nodes:
        group, artifact = node['coordinate'].split(':')
        wanted = expected(group, artifact)
        if wanted is not None and node['version'] != wanted:
            failures.append(f"{node['coordinate']}:{node['version']} expected {wanted}")
    if failures:
        raise ValueError(f'{label}: ' + '; '.join(failures))


def run(command, directory, log):
    with log.open('w') as output:
        result = subprocess.run(command, cwd=directory, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{command[0]} failed; see {log}')


with tempfile.TemporaryDirectory(prefix='published-consumers-') as temporary:
    scratch = Path(temporary)
    # One fresh cache per run; consumers are sequential in Maven to avoid concurrent local-cache writes.
    maven_nodes = {}
    for artifact in modules:
        if args.repository:
            staged = args.repository / publications[artifact]['group'].replace('.', '/') / artifact / version
            for extension in ('pom', 'module', 'jar'):
                if not (staged / f'{artifact}-{version}.{extension}').is_file():
                    raise FileNotFoundError(staged / f'{artifact}-{version}.{extension}')
        case = out / artifact
        case.mkdir(exist_ok=True)
        project = ET.Element('project', xmlns='http://maven.apache.org/POM/4.0.0')
        for key, value in [('modelVersion', '4.0.0'), ('groupId', 'release.consumer'),
                           ('artifactId', artifact + '-consumer'), ('version', '1')]:
            ET.SubElement(project, key).text = value
        repo = ET.SubElement(ET.SubElement(project, 'repositories'), 'repository')
        ET.SubElement(repo, 'id').text = 'candidate'
        ET.SubElement(repo, 'url').text = repository
        dep = ET.SubElement(ET.SubElement(project, 'dependencies'), 'dependency')
        for key, value in [('groupId', publications[artifact]['group']), ('artifactId', artifact), ('version', version)]:
            ET.SubElement(dep, key).text = value
        plugin = ET.SubElement(ET.SubElement(ET.SubElement(project, 'build'), 'plugins'), 'plugin')
        for key, value in [('groupId', 'org.apache.maven.plugins'), ('artifactId', 'maven-enforcer-plugin'), ('version', '3.6.2')]:
            ET.SubElement(plugin, key).text = value
        rules = ET.SubElement(ET.SubElement(plugin, 'configuration'), 'rules')
        includes = ET.SubElement(ET.SubElement(rules, 'dependencyConvergence'), 'includes')
        for pattern in ('com.integrallis:*', 'org.modeljars:*', 'org.modeljars.composite:*'):
            ET.SubElement(includes, 'include').text = pattern
        pom = case / 'pom.xml'
        ET.ElementTree(project).write(pom, encoding='utf-8', xml_declaration=True)
        tree = case / 'maven-graph.json'
        tree.unlink(missing_ok=True)
        run(['mvn', '-B', '-f', str(pom), f'-Dmaven.repo.local={scratch}/maven-cache',
             'enforcer:enforce', 'org.apache.maven.plugins:maven-dependency-plugin:3.8.1:tree',
             '-Dscope=runtime', '-DoutputType=json', f'-DoutputFile={tree}'], root, case / 'maven.log')
        nodes = []
        def visit(node):
            for child in node.get('children', []):
                nodes.append({'coordinate': child['groupId'] + ':' + child['artifactId'], 'version': child['version']})
                visit(child)
        visit(json.loads(tree.read_text()))
        validate(nodes, artifact + '/Maven')
        maven_nodes[artifact] = nodes
        print(f'{artifact}: Maven PASS', flush=True)

    consumer = scratch / 'gradle-consumer'
    consumer.mkdir()
    (consumer / 'settings.gradle').write_text("rootProject.name = 'published-consumers'\n" +
        '\n'.join(f"include '{module}'" for module in modules) + '\n')
    (consumer / 'build.gradle').write_text('')
    exclusive_modules = '; '.join(f"includeModule('{publications[module]['group']}', '{module}')" for module in modules)
    for artifact in modules:
        case = consumer / artifact
        case.mkdir()
        # Neither the checked repository nor its policy script is part of this build.
        (case / 'build.gradle').write_text(f"""
import groovy.json.JsonOutput
plugins {{ id 'java' }}
repositories {{
    exclusiveContent {{ forRepository {{ maven {{ url = uri('{repository}') }} }}
        filter {{ {exclusive_modules} }} }}
    mavenCentral()
}}
dependencies {{ implementation '{publications[artifact]['group']}:{artifact}:{version}' }}
tasks.register('verifyConsumer') {{
    doLast {{
        def config = configurations.runtimeClasspath
        def files = config.resolvedConfiguration.resolvedArtifacts.collect {{
            [coordinate: it.moduleVersion.id.group + ':' + it.name,
             version: it.moduleVersion.id.version, file: it.file.absolutePath]
        }}
        file('{(out / artifact / 'gradle-graph.json').as_posix()}').text = JsonOutput.toJson(files)
    }}
}}
""")
    for artifact in modules:
        (out / artifact / 'gradle-graph.json').unlink(missing_ok=True)
    run([str(root / 'gradlew'), '-p', str(consumer), '-g', str(scratch / 'gradle-cache'),
         '--no-daemon', '--console=plain', '--max-workers=4', 'verifyConsumer'], root, out / 'gradle.log')
    results = {}
    for artifact in modules:
        nodes = json.loads((out / artifact / 'gradle-graph.json').read_text())
        validate(nodes, artifact + '/Gradle')
        for node in nodes:
            if node['coordinate'].startswith(('com.integrallis:', 'org.modeljars:', 'org.modeljars.composite:')):
                jar_path = Path(node['file'])
                with zipfile.ZipFile(jar_path) as jar:
                    manifest = json.loads(jar.read('META-INF/jvm-ai/release-dependencies.json'))
                if manifest['externalPolicy'] != policy or any(
                        receipt['versions'].get(key) != value for key, value in manifest['versions'].items()):
                    raise ValueError(f'{artifact}: incompatible manifest in {node["coordinate"]}')
                node['sha256'] = hashlib.sha256(jar_path.read_bytes()).hexdigest()
        def coordinates(graph):
            return {(node['coordinate'], node['version']) for node in graph}
        if coordinates(nodes) != coordinates(maven_nodes[artifact]):
            raise ValueError(f'{artifact}: Maven and Gradle runtime graphs differ: '
                             f'{coordinates(nodes) ^ coordinates(maven_nodes[artifact])}')
        results[artifact] = {'maven': maven_nodes[artifact], 'gradle': nodes}
        print(f'{artifact}: Gradle PASS; Maven graph matches', flush=True)
    (out / 'results.json').write_text(json.dumps({'versions': receipt['versions'],
        'externalPolicy': policy, 'repository': repository, 'modules': results}, indent=2) + '\n')
