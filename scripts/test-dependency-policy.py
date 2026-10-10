#!/usr/bin/env python3
"""Exercise release dependency guards with real, offline Gradle resolution."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

root = Path(__file__).resolve().parent.parent
policy = json.loads((root / 'gradle/dependency-policy.json').read_text())


def artifact(repo, group, name, version, *, manifest=None, packaging='jar', dependencies=''):
    directory = repo / group.replace('.', '/') / name / version
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{name}-{version}.pom').write_text(
        '<project><modelVersion>4.0.0</modelVersion>'
        f'<groupId>{group}</groupId><artifactId>{name}</artifactId><version>{version}</version>'
        f'<packaging>{packaging}</packaging>{dependencies}</project>')
    if packaging == 'jar':
        with zipfile.ZipFile(directory / f'{name}-{version}.jar', 'w') as jar:
            if manifest is not None:
                jar.writestr('META-INF/jvm-ai/release-dependencies.json', json.dumps(manifest))


cases = {
    'baseline': ('', '', None),
    'stale-request': ("implementation 'com.integrallis:vectors-core:1.2.2'", '', 'requests com.integrallis:vectors-core:1.2.2'),
    'forced-downgrade': ('', "configurations.all { resolutionStrategy.force 'com.integrallis:vectors-core:1.2.2' }", 'resolved 1.2.2, expected 1.2.3'),
    'dynamic-request': ("implementation 'com.integrallis:vectors-core:1.+'", '', 'requests non-release'),
    'reverse-transitive': ("implementation 'example:bridge:1.0.0'", '', 'reverse dependency'),
    'unresolved': ("implementation 'com.integrallis:vectors-missing:1.2.3'", '', 'unresolved'),
    'project-substitution': ('', "configurations.all { resolutionStrategy.dependencySubstitution { substitute module('com.integrallis:vectors-core') using project(':local') } }", 'was substituted by a project'),
    'test-only-upgrade': ("testImplementation 'org.slf4j:slf4j-api:9.0.0'", '', 'policy requires'),
    'module-policy-downgrade': (f"implementation 'commons-codec:commons-codec:{policy['modules']['commons-codec:commons-codec']}'",
        "configurations.all { resolutionStrategy.force 'commons-codec:commons-codec:1.21.0' }",
        f"commons-codec:commons-codec resolved 1.21.0, policy requires {policy['modules']['commons-codec:commons-codec']}"),
    'missing-manifest': ("implementation 'com.integrallis:vectors-missing-metadata:1.2.3'", '', 'missing released dependency manifest'),
    'wrong-policy': ("implementation 'com.integrallis:vectors-wrong-policy:1.2.3'", '', 'upstream external dependency policy differs'),
    'framework-baseline': ("compileOnly 'org.springframework:spring-core:1.0.0'", '', None),
}
with tempfile.TemporaryDirectory(prefix='release-policy-test-') as temporary:
    base = Path(temporary)
    repo = base / 'repository'
    manifest = dict(schemaVersion=1, family='vectors', versions={'vectors': '1.2.3'}, externalPolicy=policy)
    for version in ('1.2.2', '1.2.3'):
        artifact(repo, 'com.integrallis', 'vectors-core', version,
                 manifest={**manifest, 'versions': {'vectors': version}})
    (repo / 'com/integrallis/vectors-core/maven-metadata.xml').write_text(
        '<metadata><groupId>com.integrallis</groupId><artifactId>vectors-core</artifactId>'
        '<versioning><versions><version>1.2.2</version><version>1.2.3</version></versions>'
        '</versioning></metadata>')
    artifact(repo, 'com.integrallis', 'vectors-missing-metadata', '1.2.3')
    artifact(repo, 'com.integrallis', 'vectors-wrong-policy', '1.2.3',
             manifest={**manifest, 'externalPolicy': {**policy, 'slf4j': '0.0.1'}})
    artifact(repo, 'org.modeljars', 'modeljars', '1.0.0')
    artifact(repo, 'example', 'bridge', '1.0.0', dependencies=(
        '<dependencies><dependency><groupId>org.modeljars</groupId><artifactId>modeljars</artifactId>'
        '<version>1.0.0</version></dependency></dependencies>'))
    artifact(repo, 'org.slf4j', 'slf4j-api', '9.0.0')
    for codec_version in ('1.21.0', policy['modules']['commons-codec:commons-codec']):
        artifact(repo, 'commons-codec', 'commons-codec', codec_version)
    artifact(repo, 'org.springframework', 'spring-core', '1.0.0')
    artifact(repo, 'com.fasterxml.jackson', 'jackson-bom', policy['jacksonBom'], packaging='pom')
    artifact(repo, 'org.slf4j', 'slf4j-bom', policy['slf4j'], packaging='pom')
    for name, (dependency, configuration, expected) in cases.items():
        project = base / name
        (project / 'gradle').mkdir(parents=True)
        shutil.copy(root / 'gradle/dependency-policy.json', project / 'gradle')
        shutil.copy(root / 'gradle/dependency-policy.gradle', project / 'gradle')
        (project / 'settings.gradle').write_text("rootProject.name = 'fixture'\ninclude 'local'\n")
        (project / 'local').mkdir()
        (project / 'local/build.gradle').write_text("plugins { id 'java-library' }\n")
        (project / 'build.gradle').write_text(f"""
plugins {{ id 'java-library'; id 'maven-publish' }}
group = 'com.integrallis'
version = '3.4.5'
repositories {{ maven {{ url = uri('{repo.as_uri()}') }} }}
dependencies {{ implementation 'com.integrallis:vectors-core:1.2.3'; {dependency} }}
{configuration}
ext.javaAIFamily = 'models'
ext.javaAIReleaseVersions = [vectors: '1.2.3', models: '3.4.5']
ext.javaAIPublishedProjects = [project]
apply from: 'gradle/dependency-policy.gradle'
""")
        run = subprocess.run([str(root / 'gradlew'), '-p', str(project), '--offline',
                              '--console=plain', '--max-workers=2', 'verifyReleaseTrain'],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        if (expected is None and run.returncode != 0) or (
                expected is not None and (run.returncode == 0 or expected not in run.stdout)):
            raise AssertionError(f'{name}: expected {expected!r}\n{run.stdout}')
        print(f'{name}: PASS', flush=True)
