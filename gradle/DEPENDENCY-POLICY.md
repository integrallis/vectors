# Release dependency policy

Native CI builds and invited previews may use the existing immutable
`<version>-ci.<run>.<attempt>.<12-character commit>` or
`<version>-preview.<run>.<attempt>.<12-character commit>` version for their own artifacts. Their
upstream dependencies still require exact stable releases. `verifyReleaseTrain` rejects both private
formats, and publication permits them only to the HTTPS GitHub Packages repository; public/staging
repositories require stable versions. Private verification therefore retains the public release guard.

`dependency-policy.json` is the reviewed external policy for Vectors → Models → ModelJars.
Every published Java library embeds this policy and its exact internal versions in
`META-INF/jvm-ai/release-dependencies.json`. A downstream build validates the actual upstream jars.
A sibling checkout cannot substitute for the released artifact.

`check` and staging publication verify compile, runtime and test-runtime graphs in every published
module. `verifyReleaseTrain` collects the graphs, requested/selected versions and selection reasons.
Vectors/Models `complianceCheck` and ModelJars `verifyReleaseBundle` additionally stage every
publication and run `verifyPublishedConsumerGraph`:
independent Maven and Gradle projects, fresh caches, no build-only policy injection. Internal Maven
dependency convergence and governed resolved versions must pass, and the two runtime graphs must
agree. Vectors and Models release workflows repeat consumer verification against Central after
deployment. For ModelJars, run `python3 scripts/verify-maven-runtime.py --central --output
build/central-consumers` after the USER_MANAGED Central deployment is published.

Run `python3 scripts/test-dependency-policy.py` to prove the guards reject stale requests, forced
upstream downgrades, dynamic versions, indirect reverse dependencies, unresolved artifacts, project
substitution, test-only upgrades, and absent or inconsistent upstream manifests. Its framework
baseline fixture verifies that unrelated compile-only compatibility dependencies remain allowed.
`python3 scripts/test-published-consumers.py` verifies matching Maven/Gradle metadata, then changes
only the POM: Gradle must still resolve the expected version and the Maven consumer must reject it.

## Reviewed selection, 2026-10-10

| Dependency | Selection and basis |
| --- | --- |
| Jackson 2 | BOM 2.22.3; annotations 2.22 as specified by that BOM. [Release notes](https://github.com/FasterXML/jackson/wiki/Jackson-Release-2.22.3). Jackson 3 uses different packages and requires a separate migration. |
| SLF4J | Stable 2.0.20. The metadata's 2.1.0-alpha1 is a prerelease. [Publisher news](https://www.slf4j.org/news.html). |
| Netty | 4.1.139.Final security patch for the supported 4.1 line. [Release notes](https://netty.io/news/2026/10/06/4-1-139-Final.html). 4.2 is a separate transport migration; 4.1 support ends July 2027. |
| AWS SDK | 2.55.14, [published 2026-10-09](https://github.com/aws/aws-sdk-java-v2/releases/tag/2.55.14). S3 integration tests must cover the changed checksum defaults introduced after the old 2.29 baseline. |
| Caffeine | 3.3.0, including its matching JCache integration. [Release notes](https://github.com/ben-manes/caffeine/releases/tag/v3.3.0). |
| Avaje JSON | 3.15, the latest verifiable source tag and release notes. Central advertises 3.16, but neither source tag nor release notes were available during this review; defer that artifact pending provenance review. [Releases](https://github.com/avaje/avaje-jsonb/releases). |
| Arrow | 19.0.0, current Java release. [Publisher release](https://arrow.apache.org/blog/2026/03/16/arrow-java-19.0.0/). |
| Commons Codec | 1.22.1; encoding and decoding corrections since Arrow's transitive 1.21.0. [Release notes](https://commons.apache.org/proper/commons-codec/changes.html). |
| Apache HttpClient / HttpCore | Stable 5.6.5 / 5.4.4; connection cleanup, async decompression and interrupted-write fixes. Newer 5.7-alpha / 5.5-beta lines are prereleases. [Client notes](https://github.com/apache/httpcomponents-client/blob/rel/v5.6.5/RELEASE_NOTES.txt), [Core notes](https://github.com/apache/httpcomponents-core/blob/rel/v5.4.4/RELEASE_NOTES.txt). |
| JCache, reactive streams, FlatBuffers, AWS eventstream | Existing 1.1.1, 1.0.4, 25.2.10, 1.0.1 remain current in Maven Central. |

Spring and LangChain4j compile-only compatibility baselines are deliberate: Vectors tests Spring AI
1.1.4 / Boot 3.4.5 and LangChain4j 1.13.1; Models uses AI 2.0.0 / Boot 4.1.0 and LangChain4j 1.17.2.
They are not forced to a common major merely to align version strings. Applications must select one
compatible framework profile; the shared JSON and logging policy is still checked in those tests.

Changing a governed version requires regenerating locks with `--write-locks`, exercising production
and adapter tests, staging publications, and validating both clean consumers. Updating this document
or a version property alone does not establish compatibility or absence of vulnerabilities.

Individual module constraints are ordinary published constraints. Where Maven's transitive mediation
would retain an older dependency, the optional integration also declares that dependency directly.
The clean consumers verify the result. The core library does not gain optional Arrow or HTTP runtime
dependencies merely because the policy constrains their versions when present.
