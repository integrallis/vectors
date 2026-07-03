# Contributing to java-vectors

Thank you for your interest in contributing to java-vectors! This guide will help you get started.

## Development Setup

### Prerequisites

- **JDK 25** — The Vector API (incubating) and mature FFM require JDK 25+
- **Gradle 9.4.1** — Included via the Gradle Wrapper (`./gradlew`)
- **Git** — For version control

### Clone and Build

```bash
git clone https://github.com/integrallis/vectors.git
cd vectors
./gradlew build -x :docs:build
```

The first build downloads dependencies and compiles all modules. The `-x :docs:build` flag skips documentation generation for faster iteration.

## Build Commands

All commands run from the `vectors/` directory:

```bash
# Full build (all library modules)
./gradlew build -x :docs:build

# Run all tests (excludes slow/benchmark/integration)
./gradlew test

# Run a single test class
./gradlew :vectors-core:test --tests "com.integrallis.vectors.core.SomeTest"

# Unit tests only
./gradlew unitTest

# Slow tests (large datasets)
./gradlew slowTest

# Code formatting (auto-fix)
./gradlew spotlessApply

# Code formatting (check only)
./gradlew spotlessCheck

# Compliance checks (SBOM, governance, reproducibility)
./gradlew complianceCheck

# Generate documentation
./gradlew :docs:build
./gradlew aggregateJavadoc
```

## Test Conventions

We follow **test-driven development** with integration tests first:

1. Write an integration test exercising the public API end-to-end
2. Run it — it fails (Red)
3. Implement the minimum code to make it pass (Green)
4. Refactor, then add `@Tag("unit")` tests for edge cases
5. Repeat

### Test Tags

| Tag           | Task                | Description                              |
|---------------|---------------------|------------------------------------------|
| *(default)*   | `test`              | All tests except slow/benchmark/infra    |
| `unit`        | `unitTest`          | Fine-grained unit tests                  |
| `slow`        | `slowTest`          | Large dataset tests                      |
| `benchmark`   | —                   | Performance regression tests             |
| `integration` | `integrationTest`   | Docker-dependent (TestContainers)        |
| `distributed` | `distributedTest`   | Multi-node cluster tests                 |
| `k8s`         | `k8sTest`           | Kubernetes deployment tests              |
| `chaos`       | `chaosTest`         | Chaos engineering / fault injection      |
| `scale`       | `scaleTest`         | Large-scale scalability tests            |

Always add the appropriate `@Tag` annotation to new test classes.

## Code Style

- **Google Java Format** — Enforced via Spotless. Run `./gradlew spotlessApply` before committing.
- **No Lombok** — This is a low-level library; we use explicit code.
- **Minimal dependencies** — Library modules should only depend on `slf4j-api` at runtime.
- **Java-first CPU release** — The 0.1.x artifacts require no JNI library.
  Storage may call `posix_madvise` through FFM as an optional optimization.
  Experimental GPU code and its external native runtime are outside the
  published artifact set.

## Module Structure

```
vectors-core/          — SIMD distance kernels, vector types
vectors-storage/       — Off-heap memory, mmap, arena storage
vectors-quantization/  — SQ, PQ, BQ, NVQ, RaBitQ quantizers
vectors-hnsw/          — HNSW graph index
vectors-vamana/        — Vamana/DiskANN graph index
vectors-ivf/           — IVF family indexes
vectors-db/            — High-level vector collection API
vectors-bench/         — JMH benchmarks
```

Dependencies flow downward: `core -> storage -> quantization -> {hnsw, vamana, ivf} -> db`.

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`
2. **Write tests first** — follow the TDD workflow described above
3. **Keep changes focused** — one feature or fix per PR
4. **Run the full build** before submitting:
   ```bash
   ./gradlew spotlessCheck build :vectors-bench:recallGate complianceCheck -x :docs:build
   ```
5. **Write a clear PR description** — explain what changed and why
6. **CI must pass** — the GitHub Actions pipeline runs build + test + SBOM generation

## Reporting Issues

- Use [GitHub Issues](https://github.com/integrallis/java-vectors/issues) for bugs and feature requests
- For security vulnerabilities, see [SECURITY.md](SECURITY.md)
