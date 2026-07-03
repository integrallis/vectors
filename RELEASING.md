# Releasing java-vectors

The release path mirrors the proven `integrallis/mfcqi-java` pipeline: Gradle
stages Maven publications, JReleaser signs and validates one bundle, and the
Central Publisher Portal publishes it.

## Release scope

Version `0.1.x` publishes the Apache-licensed CPU libraries and adapters defined
by `publishedProjects` in `build.gradle.kts`. FSL modules, applications, Studio,
benchmarks, the optimizer, and the server client are deliberately excluded from
Maven Central until they have independent release contracts.

## Cut a release

1. Set a non-snapshot version in `gradle.properties`.
2. Update `CHANGELOG.md`, commit, and push to `main`.
3. Run **Actions → Release** with `dry_run` enabled.
4. Inspect the JReleaser validation output and staged POMs.
5. Run **Release** again with `dry_run` disabled.

Maven Central versions are immutable. Use `skip_deploy` only when Central
already succeeded and the GitHub release needs to be recreated.

## Required secrets

- `MAVENCENTRAL_USERNAME`
- `MAVENCENTRAL_PASSWORD`
- `GPG_PUBLIC_KEY`
- `GPG_SECRET_KEY`
- `GPG_PASSPHRASE`

These are the same secret names used by `mfcqi-java`. The `com.integrallis`
namespace and GPG identity must already be configured in Central.

## Local staging

```bash
./gradlew clean spotlessCheck build :vectors-bench:recallGate complianceCheck publish \
  -x :docs:build
```

Artifacts are written to `build/staging-deploy`. A real release must be built
from a clean commit with a non-`SNAPSHOT` version.
