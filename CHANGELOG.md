# Changelog

All notable changes to java-vectors are documented here.

## [Unreleased]

### Added

- Maven Central staging and JReleaser release automation based on the proven
  `integrallis/mfcqi-java` pipeline.
- Explicit 0.1.x publication allowlist, strict staged-artifact validation,
  CycloneDX SBOM validation, dependency locks, and an enforced 80% instruction
  coverage gate for every published module.
- Regression coverage for scalar kernels, fused similarity, core value types,
  HNSW compaction/merge, and every LangChain4j cache-store mutation overload.
- A claims-controlled JVM community launch brief and release-day demonstration
  contract.

### Fixed

- Removed the optional FSL-licensed GPU module from `vectors-db`'s transitive
  runtime dependency graph.
- Corrected CI paths for the standalone repository layout.
- Corrected the `vectors-cluster` license inventory.
- Removed stale Sigstore dependencies and regenerated every module lockfile
  without swallowing resolution failures.
- Aligned SLF4J on 2.0.17 so uncached SpotBugs analysis resolves logging
  classes instead of silently reporting an incomplete auxiliary classpath.
- Replaced native page-size discovery during class initialization with a
  portable logical format alignment.
- Corrected stale API documentation and made Javadoc errors fail the build.
- Stabilized the persistent-reader concurrency regression under the full suite.
- Changed the VCR end-to-end demo to strict playback, migrated its cassettes to
  the current framed local-storage format, and proved default tests do not
  rewrite tracked source fixtures.

### Changed

- Rewrote release-facing documentation around the implemented single-process
  CPU scope and removed unsupported performance, scale, distributed, and GPU
  claims.
