# Security Policy

## Supported Versions

| Version            | Status                         |
|--------------------|--------------------------------|
| 0.1.x (unreleased) | Pre-release security fixes only |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in java-vectors, please report it responsibly.

### How to Report

**Email:** security@integrallis.com

Please include:
- A description of the vulnerability
- Steps to reproduce the issue
- The potential impact
- Any suggested fix (optional)

### Response Timeline

| Action                     | Timeline         |
|----------------------------|------------------|
| Acknowledgment of report   | Within 48 hours  |
| Initial assessment         | Within 5 days    |
| Fix development            | Within 30 days   |
| Public disclosure           | Within 90 days   |

### Coordinated Disclosure

We follow a **90-day coordinated disclosure policy**:

1. Reporter submits vulnerability via email
2. We acknowledge receipt within 48 hours
3. We assess severity and develop a fix
4. We release a patched version
5. We publish a security advisory (GitHub Security Advisories)
6. Reporter may publish their findings after 90 days or after the fix is released, whichever comes first

### What Qualifies

- Memory safety issues (buffer overflows, use-after-free in Arena-managed segments)
- Denial of service via crafted vector data or index structures
- Information disclosure through side channels
- Any bug that compromises data integrity of stored vectors

### What Does NOT Qualify

- Performance issues or inefficiencies (report as regular issues)
- Bugs in test code or benchmarks
- Issues requiring physical access to the machine

## Security-Relevant Design Properties

Security-relevant boundaries:

- **CPU release artifacts are Java bytecode** — They require no JNI library. The storage module
  uses the FFM API for mmap access and an optional `posix_madvise` call.
- **Native and network modules exist but are excluded from 0.1.x** — `vectors-gpu` binds to cuVS;
  distributed, server, replication, and S3-capable modules perform network I/O. They must not be
  described as part of the CPU release boundary.
- **Arena-based memory management** — Off-heap segments use `java.lang.foreign.Arena`; callers must
  still respect the documented lifetime and ownership contracts.
- **Artifact signing is external** — JReleaser signs staged Maven artifacts with the maintainer's
  GPG key. Runtime libraries do not perform release-signing operations.
- **Dependencies vary by module** — Review each published POM and generated SBOM; framework
  adapters intentionally depend on their framework APIs.

## Supply Chain Security

- **SBOM generation** — CycloneDX SBOMs for every published module are generated and validated by
  `complianceCheck`
- **Dependency locking** — Gradle lockfiles pin all transitive dependency versions
- **Vulnerability scanning** — OWASP Dependency-Check is available as an explicit release audit;
  findings with CVSS >= 7.0 fail that task
- **Artifact signing** — The release workflow is configured to sign Maven
  Central artifacts with GPG through JReleaser; no 0.1.0 release has been
  signed yet.
- **Deterministic JAR settings** — JAR tasks disable file timestamps and use
  reproducible file order. A byte-for-byte clean-room rebuild has not yet been
  independently verified.
- **Code quality** — An MFCQI (Multi-Factor Code Quality Index) workflow scores
  the library on pull requests and pushes to `main`. Static security signal is
  provided by OpenSSF Scorecard and the MFCQI security metric; a dedicated SAST
  scanner is not currently wired.
