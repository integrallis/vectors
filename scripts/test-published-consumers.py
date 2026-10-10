#!/usr/bin/env python3
"""Prove consumer verification catches a POM-only drift hidden by correct Gradle metadata."""
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

root = Path(__file__).resolve().parent.parent
policy = json.loads((root / 'gradle/dependency-policy.json').read_text())
with tempfile.TemporaryDirectory(prefix='publication-policy-test-') as temporary:
    base = Path(temporary)
    repo = base / 'repository'
    directory = repo / 'com/integrallis/vectors-fixture/9.9.9'
    directory.mkdir(parents=True)
    jar_name = 'vectors-fixture-9.9.9.jar'
    manifest = {'schemaVersion': 1, 'family': 'vectors', 'versions': {'vectors': '9.9.9'}, 'externalPolicy': policy}
    with zipfile.ZipFile(directory / jar_name, 'w') as jar:
        jar.writestr('META-INF/jvm-ai/release-dependencies.json', json.dumps(manifest))
    pom = directory / 'vectors-fixture-9.9.9.pom'
    def write_pom(slf4j):
        pom.write_text(f'''<project><modelVersion>4.0.0</modelVersion>
<!-- do_not_remove: published-with-gradle-metadata -->
<groupId>com.integrallis</groupId><artifactId>vectors-fixture</artifactId><version>9.9.9</version>
<dependencies><dependency><groupId>org.slf4j</groupId><artifactId>slf4j-api</artifactId>
<version>{slf4j}</version></dependency></dependencies></project>''')
    write_pom(policy['slf4j'])
    (directory / 'vectors-fixture-9.9.9.module').write_text(json.dumps({
        'formatVersion': '1.1',
        'component': {'group': 'com.integrallis', 'module': 'vectors-fixture', 'version': '9.9.9'},
        'variants': [{'name': 'runtimeElements', 'attributes': {'org.gradle.usage': 'java-runtime',
            'org.gradle.category': 'library', 'org.gradle.libraryelements': 'jar'},
            'files': [{'name': jar_name, 'url': jar_name}],
            'dependencies': [{'group': 'org.slf4j', 'module': 'slf4j-api', 'version': {'requires': policy['slf4j']}}]}]}))
    receipt = base / 'train.json'
    receipt.write_text(json.dumps({**manifest, 'modules': [{'project': ':vectors-fixture', 'failures': [],
        'publication': {'group': 'com.integrallis', 'artifact': 'vectors-fixture', 'version': '9.9.9'}}]}))
    command = ['python3', str(root / 'scripts/verify-maven-runtime.py'), '--repository', str(repo),
               '--receipt', str(receipt), '--output', str(base / 'output')]
    first = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if first.returncode:
        # Keep the useful nested resolver error before the fixture directory is removed.
        for log in (base / 'output').rglob('*.log'):
            print(log.read_text()[-6000:])
        raise AssertionError(first.stdout)
    print('matching POM and Gradle metadata: PASS', flush=True)
    write_pom('2.0.17')
    # Prove Gradle still sees the correct graph after only the POM is changed.
    gradle = base / 'metadata-consumer'
    gradle.mkdir()
    (gradle / 'settings.gradle').write_text("rootProject.name = 'metadata-consumer'\n")
    (gradle / 'build.gradle').write_text(f'''
plugins {{ id 'java' }}
repositories {{
    maven {{ url = uri('{repo.as_uri()}') }}
    mavenCentral()
}}
dependencies {{ implementation 'com.integrallis:vectors-fixture:9.9.9' }}
tasks.register('verifyMetadata') {{ doLast {{
    def slf4j = configurations.runtimeClasspath.resolvedConfiguration.resolvedArtifacts.find {{
        it.moduleVersion.id.group == 'org.slf4j' && it.name == 'slf4j-api'
    }}
    assert slf4j.moduleVersion.id.version == '{policy['slf4j']}'
}} }}
''')
    metadata = subprocess.run([str(root / 'gradlew'), '-p', str(gradle), '--no-daemon',
        '--console=plain', 'verifyMetadata'], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if metadata.returncode:
        raise AssertionError('Unchanged Gradle metadata did not retain the correct version:\n' + metadata.stdout)
    second = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if second.returncode == 0 or 'org.slf4j:slf4j-api:2.0.17 expected' not in second.stdout:
        raise AssertionError('POM-only downgrade was not rejected for the expected reason:\n' + second.stdout)
    print('POM-only downgrade with unchanged correct Gradle metadata: REJECTED', flush=True)
