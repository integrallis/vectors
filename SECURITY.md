# Security Policy

## Supported Versions

| Version     | Supported          |
|-------------|--------------------|
| 0.1.x       | :white_check_mark: |

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

java-vectors is designed with security in mind:

- **Pure Java** — No JNI, no FFM bindings to native code, no native backends. The entire codebase is Java, eliminating classes of native memory corruption bugs.
- **No cryptographic operations** — java-vectors does not implement or depend on cryptographic primitives. Artifact signing uses Sigstore (external tooling), not library code.
- **Arena-based memory management** — All off-heap memory is managed through `java.lang.foreign.Arena`, providing spatial and temporal safety guarantees enforced by the JVM.
- **No network I/O** — The core library modules perform no network operations. All data is local (in-memory or mmap).
- **Minimal dependencies** — Library modules depend only on `slf4j-api` at runtime, minimizing supply chain attack surface.

## Supply Chain Security

- **SBOM generation** — CycloneDX SBOMs are generated for every module on every build
- **Dependency locking** — Gradle lockfiles pin all transitive dependency versions
- **Vulnerability scanning** — OWASP Dependency-Check scans for known CVEs (CVSS >= 7.0 fails the build)
- **Artifact signing** — Release artifacts are signed with Sigstore for tamper-evident distribution
- **Reproducible builds** — JAR artifacts are reproducible (deterministic timestamps and file ordering)
- **SAST** — GitHub CodeQL runs on every pull request
