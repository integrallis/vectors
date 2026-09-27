# Vectors 0.1.24 release qualification

Candidate: embedding provenance (PR #78, merged as `5dc5835f`). **This release breaks the on-disk
format**, which sets the bar for what must be shown.

## What is being released

`EmbeddingRecipe`, `ContentHash`, the `RecipeCodec` SPI with a dependency-free default, recipe
persistence anchored in the manifest, Studio provenance surfacing, prefix application in the Spring AI
adapter, and one new published module, `vectors-db-jackson`.

## Bounded infrastructure preflight

No paid hardware required. Every gate below is satisfied by the CI matrix on the candidate commit plus
local runs; nothing here needs a VPS or a GPU.

## Gates

Numbered to match 0.1.23 so the bar is visibly the same one.

| # | gate | evidence | status |
|---|---|---|---|
| 1 | x86 clean build, formatting, recall regression, compliance, dependency locks, workflow validation; stage publications and inspect coordinates, license and dependencies | CI `build` (Compliance Check) on the candidate; staged POMs from the release dry run | **pass** (CI); POM inspection pending the dry run |
| 2 | Q6_K/Q4_K bit-equality and cold/hot JIT tests; core tests rerun with `vectors.maxBits=128` and the forced scalar provider in forked JVMs | CI `Core portability (x86-128)`, `Core portability (x86-scalar)` | **pass** |
| 3 | **ARM: core tests on real AArch64 Java 25**, including the Q6_K contracts. A forced 128-bit x86 run is not evidence of ARM execution | CI `Core portability (arm64)` | **pass** |
| 4 | Notebooks against the staged artifacts, documentation build, SBOM | CI `Execute Java Notebooks`; `verifyDocumentation` locally; `cyclonedxDirectBom` per module in CI | **pass** |
| 5 | Studio browser workflows, integration tests and UI demos against the candidate | CI `Studio browser workflows` | **pass** |
| 6 | Ready only after the above pass **on the exact candidate source** | all 13 CI jobs green on `05e9f44f`, merged unchanged as `5dc5835f` | **pass** |

Additional gates this release requires, because of what it changes:

| # | gate | evidence | status |
|---|---|---|---|
| 7 | Framework compatibility across every supported version, since both adapters changed | CI: Spring AI 1.0.0 / 1.1.8 / 2.0.0, LangChain4j 1.0.0 / 1.13.1 / 1.17.1 | **pass** |
| 8 | The new published coordinate `vectors-db-jackson` resolves with the right POM: `vectors-core` as API, Jackson **2.21.7** matching the other published modules that ship it | generated POM inspected locally; staged POM pending the dry run | **partial** |
| 9 | Recipe persistence behaves under corruption, absence, orphaning and a newer schema | 9 `RecipeStoreTest` cases, 7 `EmbeddingRecipeLifecycleTest` cases through a real collection | **pass** |
| 10 | Any `RecipeCodec` meets its four documented obligations | 7 `RecipeCodecContractTest` cases | **pass** |

Local suites on the candidate: **2480 tests, zero failures** across core, db, db-jackson, spring-ai and
langchain4j.

## What is explicitly not claimed

- **No performance claim.** ARM *correctness* for the pinned reductions is evidenced by gate 3; ARM
  *throughput* is unmeasured, and the pinned 8-lane species is emulated on 128-bit hardware. If it
  regresses, the remedy is a fixed logical blocking built from two 128-bit vectors, never a return to
  host-dependent results.
- **No retrieval-quality claim.** The Spring AI prefix fix corrects a documented model contract that was
  not being honoured. Nobody has measured what it is worth, and "we improved retrieval" is not a
  statement this project is entitled to make without a benchmark on a public dataset.
- **Provenance is drift and accident protection, not a signature.** Anyone able to write both the
  sidecar and the manifest can recompute the hash and the CRC. Tamper-evidence would need a signed
  manifest, which is a separate feature, and `ATTESTED` must not be read as implying it.

## Migration

There is none. `VERSION_MANIFEST` 5 refuses version-4 collections and the error names both versions and
says to rebuild. Accepted on the basis that the library is newly released with no known production
deployments. Anyone holding a 0.1.23 collection must rebuild it or export with 0.1.23 first.

## Downstream

`models` pins `vectorsVersion = 0.1.23` and compiles clean against it; nothing in this release breaks
it, since the `Document` change kept the four-argument form. Adoption is a separate step: `models` has
no asymmetric-prefix handling while shipping Nomic v1.5 and E5-Mistral, recorded in
`models/docs/findings/asymmetric-embedding-prefixes.md`. That bump must identify the immutable released
version.
