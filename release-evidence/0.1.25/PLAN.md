# Vectors 0.1.25 release qualification

Candidate: in-place manifest migration (PR #81). **This release exists because 0.1.24 shipped a
mandatory hard stop with no way through it**, which sets the bar for what must be shown.

## Why this release

0.1.24 moved `VERSION_MANIFEST` 4 → 5 with an exact-match check, so a 0.1.23 collection will not open.
That was intended and documented. Three things about it were not acceptable:

1. The error blamed the filesystem — `generation directory already exists` — because recovery
   bootstrapped over the unreadable generation before anything checked the version. The fix was merged
   at 12:39; the 0.1.24 tag was cut at 10:49, so **it shipped without it**.
2. The documented remedy was to rebuild the collection, meaning re-run an embedding model over the
   whole corpus. This release demonstrates that was unnecessary.
3. Studio dropped older collections from its listing entirely, leaving one line in a log.

## What is being released

`ManifestMigration` with an inspect/migrate/CLI surface, `migrateOlderFormats` on the builder,
`java-vectors.migrate-on-startup` in the starter, Studio format surfacing, and two new published
modules — `vectors-spring-batch` and `vectors-jakarta-batch`.

## Bounded infrastructure preflight

No paid hardware required. Every gate below is satisfied by the CI matrix on the candidate commit plus
local runs; nothing here needs a VPS or a GPU.

## Gates

Numbered to match 0.1.24 so the bar is visibly the same one.

| # | gate | evidence | status |
|---|---|---|---|
| 1 | x86 clean build, formatting, recall regression, compliance, dependency locks, workflow validation; stage publications and inspect coordinates, license and dependencies | CI `build` (Compliance Check) on the candidate; staged POMs from the release dry run | |
| 2 | Q6_K/Q4_K bit-equality and cold/hot JIT tests; core tests rerun with `vectors.maxBits=128` and the forced scalar provider in forked JVMs | CI `Core portability (x86-128)`, `Core portability (x86-scalar)` | |
| 3 | **ARM: core tests on real AArch64 Java 25**, including the Q6_K contracts. A forced 128-bit x86 run is not evidence of ARM execution | CI `Core portability (arm64)` | |
| 4 | Notebooks against the staged artifacts, documentation build, SBOM | CI `Execute Java Notebooks`; `verifyDocumentation`; `cyclonedxDirectBom` per module | |
| 5 | Studio browser workflows, integration tests and UI demos against the candidate | CI `Studio browser workflows` | |
| 6 | Ready only after the above pass **on the exact candidate source** | all CI jobs green on the candidate, merged unchanged | |

Additional gates this release requires, because of what it changes:

| # | gate | evidence | status |
|---|---|---|---|
| 7 | **A migrated collection behaves identically to a rebuilt one**, not merely opens. Measured on a real version 4 collection, not a synthesised header | The bundled 1,929-prompt router index migrated in place, then evaluated on the 481 held-out prompts: 0.9044 (435/481) with **per-item outcomes byte-identical** to a full 15-minute rebuild | |
| 8 | Migration is safe to interrupt and safe to repeat | `ManifestMigrationTest`: previous manifest kept as `.bak` and byte-compared; tmp/fsync/ATOMIC_MOVE/dir-fsync; migrating twice is a no-op with a byte-identical manifest | |
| 9 | A corrupt header is refused, not laundered | `refusesToMigrateACorruptHeaderRatherThanLaunderIt`: a flipped payload length fails the version 4 self CRC and the migration aborts rather than recomputing a fresh CRC over bad bytes | |
| 10 | A newer manifest is never downgraded | `leavesANewerManifestAloneInsteadOfDowngradingIt`: reported and skipped, bytes unchanged | |
| 11 | Migration never runs unasked | Off in the builder, off in the starter, report-only in the CLI and both batch adapters; `refusesAnOlderCollectionByDefaultAndSaysHowToMigrateIt` pins the default | |
| 12 | A dry run cannot be mistaken for a completed migration | `Outcome.NEEDS_MIGRATION` is distinct from `MIGRATED`, asserted in both directions; the batchlet returns `PENDING`, and an unrecognised `apply` value is refused rather than read as false | |
| 13 | The two new published coordinates carry no framework dependency of their own | Spring Batch and `jakarta.batch-api` are `compileOnly`; `gradle.lockfile` per module; `verifyLockfiles` and `verifyDocumentation` green | |

## Fixture discipline

The migration fixtures are built by writing a real collection with this build and then **downgrading**
its manifests to version 4, rather than by hand-assembling a 164-byte header. A hand-made fixture
would only prove the migrator agrees with the test author's reading of the format. Gate 7 goes
further and uses a genuine 0.1.23-era artifact — the index that shipped in `models-router`.

## Local evidence

`spotlessCheck build complianceCheck test --rerun-tasks -x :docs:build`: 4148 tests, 0 test failures,
0 task failures. `:docs:build` is excluded locally only because Antora will not load a git worktree's
`.git` file; CI covers it in a normal checkout.
