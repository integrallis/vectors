# Vectors 0.1.23 qualification — 2026-09-25

Scope: the existing 30 Apache CPU publications, plus compatibility checks for Studio and the
desktop multimodal RAG demo. No throughput or general retrieval-quality improvement is claimed.
The Q6_K correction was already merged in #76; this candidate makes it releasable with refreshed
application, dependency and hardware checks. No Maven Central publication has occurred.

## Candidate and executed gates

Product code: `5d5e8bd323baef6548dcb3077ee9940bc355646b`. Commit `8166ad4` adds permanent
ARM/scalar/128-bit CI checks and documents the infrastructure substitution; product code is identical.

| Gate | Measured result |
|---|---|
| Dedicated x86 Linux full build/staging | Successful clean build, formatting, recall gate, compliance, lockfiles, workflow validation, documentation, SBOM and publication staging. Captured final XML: 4,266 tests, 0 failures/errors, 26 skips. |
| x86 core with `vectors.maxBits=128` | 1,489 tests, 0 failures/errors/skips; property passed to forked JVMs through `JAVA_TOOL_OPTIONS`. |
| x86 core with `vectors.forceScalar=true` | 1,489 tests, 0 failures/errors/skips, independently executed. |
| Real ARM core | 1,489 tests, 0 failures/errors/skips on native AArch64 Neoverse-N2; permanent CI also repeats x86-128/scalar with the same counts. |
| Docker-backed S3 storage integration | 32 tests pass against LocalStack 4.1.0, including conditional writes. |
| Studio HTTP/SSE integration | Required integration task passes on VPS and in CI. |
| Studio Chromium browser | All 11 workflows pass on VPS and CI: collections/pagination, escaped document text, empty collection, PCA, t-SNE, UMAP, neighbor/MMR controls, 2D, dataset/provider listings and collection deletion. |
| Packaged desktop application | JavaFX distribution opens; uploads the repository's seven-page `Vector_database.pdf`; local MiniLM embeddings and ONNX layout detection index 15 chunks; PDF next-page control displays page 2/7. |
| Desktop → server → Studio | Three additional browser checks pass: all 15 documents listed, extracted image blob rendered, PCA projection and neighbor lookup over the 384-dimensional embeddings. |
| Framework compatibility | CI passes the six supported Spring AI/LangChain4j compatibility configurations. |
| Published metadata | All 30 staged POMs use `com.integrallis`, version `0.1.23`, and Apache-2.0 license metadata. |
| Runtime dependency advisory check | OSV queried at 20:02:22 UTC for 63 external runtime components in the publication closure: 0 advisory matches after upgrades. This is a dated advisory check, not proof of absence of vulnerabilities. |
| Release dry run | Successful signed Maven Central deployment validation and staged-artifact notebooks; no deployment/tag/release created. |

Runs: [product-code CI](https://github.com/integrallis/vectors/actions/runs/36183314663),
[framework compatibility](https://github.com/integrallis/vectors/actions/runs/36183314662),
[release dry run](https://github.com/integrallis/vectors/actions/runs/36183212367),
[permanent hardware matrix](https://github.com/integrallis/vectors/actions/runs/36184908536).
The release dry run used `dc7b123`; subsequent product changes only fix the excluded desktop
application launcher. CI at `5d5e8bd` verifies that launcher together with the full repository.

## Findings and corrections

- The published scalar Q6_K fold disagreed with SIMD in low bits. The merged fix is included;
  the existing bit-equality and cold/hot contracts are exercised on the qualified paths.
- LocalStack 3.8 did not enforce the conditional-write contract: two stale-ETag tests failed.
  Upgrade the emulator to 4.1.0 and keep the integration task mandatory.
- UMAP on a connected graph required native ARPACK in addition to BLAS. Install it in CI and
  document it for operators; improve the missing-library error.
- Flex styling overrode `hidden` on the MMR lambda field. Make the hidden attribute effective;
  browser checks now cover the control's visibility before and after MMR selection.
- The packaged desktop launcher failed with “JavaFX runtime components are missing.” Use a
  wrapper main class and apply macOS Dock arguments only on macOS.
- Jackson/Netty advisories were present in optional published runtime dependencies. Upgrade
  Jackson to 2.21.7 and align Netty to 4.1.138.Final without changing its major version.
- Two core test assumptions prevented legitimate forced-width/scalar execution. Assert the
  configured width and exact supported fallback while retaining the integer-fold oracle where
  its SIMD path is eligible. These changes do not alter a kernel to obtain passing results.

## Evidence and limits

ARM CI ran the GitHub PR merge tree `348cdad21f723a6ed3504b244f0ba848d00ce48d` for head `8166ad4`.
The x86 host was Ubuntu 24.04, AMD EPYC Milan (8 dedicated vCPU, 32 GB), Temurin
25.0.4.1+1, with Chromium 153.0.8010.12 and Playwright 1.63.0. Browser projections use software
WebGL. The synthetic 96-document fixture tests application contracts, not retrieval/model quality.
Desktop ingestion uses the existing public PDF fixture; no paid generation provider was invoked.

The 26 full-build skips comprise 24 live-S3 credential tests, one live-R2 ingestion test and one
optional ONNX fixture test. Desktop ONNX layout detection subsequently ran successfully, but that
does not retroactively count the skipped test as executed. Provider-backed chat, live R2/S3,
browser/device combinations beyond Chromium, GPU/distributed backends and dataset downloads
are not qualified by these results. The desktop welcome text also reflects configured defaults
instead of the provider selected automatically; treat this as a remaining UI polish issue.

Initial failures are retained: root-run permission semantics, stale documentation version,
LocalStack conditional writes, missing ARPACK, core test configuration assumptions, and the
desktop launcher. The first cross-app image assertion targeted a `PAGE_RENDER` context-only
chunk; the demo explicitly excludes these from user image display. The corrected check targets
an actual `IMAGE` chunk. No product change was made to satisfy that mistaken assertion.

Local raw evidence is under `projects/benchmark-results/vectors-0.1.23-20260925/` in the parent
workspace: logs, XML, screenshots, Playwright traces, dependency queries, staged artifacts and
infrastructure records. The 1,933-file x86 manifest was verified after copying, before deletion:
`41a4fb44b03917d0288c8a62492bd5d4f5c11cf325aadddfbcbd69b07d61644d` (SHA-256 of manifest).
The x86 server and its firewall were deleted; no run-owned server or IP remains. Hetzner rejected
ARM creation, so the real ARM gate uses GitHub's native runner. The unrelated pre-existing
Granite firewall was preserved. Branch protection/rulesets were absent when inspected; green
checks here do not establish an enforced repository merge policy.

The rejected band-GEMM and dequantization experiments (#74/#75) stay out of this release.
Their branches, worktrees and negative evidence are preserved.
