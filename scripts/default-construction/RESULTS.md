# Measured default construction improvements — 2026-10-03

**No new option is required.** These are timings for the final, reduced implementation, not the
rejected SQ8 or parallel-builder variants.

| Workload | Baseline median | Candidate median | Less elapsed time |
|---|---:|---:|---:|
| Persistent COSINE collection, 100,000 × 512, 4 commits, 8 workers | 53.072 s | 49.070 s | **7.54%** |
| Serial HNSW, Wikipedia 30,000 × 768, DOT_PRODUCT | 38.175 s | 35.168 s | **7.88%** |

All six paired timings improved. The baseline is `4bc505b3e8c0c217e9d66727d643921a0d8efd55`,
which already includes the earlier batch-write fix in #85. Measured product source is commit
`eb1897510faa61e3319c3c2e811d906393c26a3b`; subsequent report/test-runner changes do not alter
those runtime sources. [Provenance](evidence/provenance.json) records source and frozen-jar hashes.

## Every timing sample

| Workload | Pair / level seed | Baseline seconds | Candidate seconds | Less time |
|---|---:|---:|---:|---:|
| collection | 0 | 53.518366 | 49.750098 | 7.04% |
| collection | 1 | 53.072274 | 48.729711 | 8.18% |
| collection | 2 | 53.054094 | 49.069997 | 7.51% |
| serial | 42 | 37.796097 | 34.309587 | 9.22% |
| serial | 43 | 38.174717 | 35.167819 | 7.88% |
| serial | 44 | 38.563893 | 35.385051 | 8.24% |

Collection order was candidate/baseline, baseline/candidate, candidate/baseline. Serial order was
baseline/candidate, candidate/baseline, baseline/candidate. Every arm used a fresh JVM. No other
benchmark or local build ran during timing. Later recall probes and tests could run concurrently;
their elapsed times are not performance claims.

## Recall and exactness

**Serial:** every pair has an identical SHA-256 over every node, level, edge ID and raw edge-score
bits. Recall@10 is identical at all six search budgets for all three seeds, using 1,000 held-out
queries and exhaustive truth for the 30,000-row prefix.

**Collections:** every paired recall difference is positive; none loses recall. These probes use
1,000 evenly spaced stored rows, exhaustive truth against the full 100,000-row prefix, and reopened
collections. They are in-corpus queries, not the held-out Wikipedia workload.

Absolute recall differences, candidate minus baseline:

| Pair | ef=10 | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | +0.0004 | +0.0004 | +0.0002 | +0.0003 | +0.0003 | +0.0004 |
| 1 | +0.0013 | +0.0005 | +0.0005 | +0.0004 | +0.0004 | +0.0004 |
| 2 | +0.0002 | +0.0005 | +0.0003 | +0.0003 | +0.0002 | +0.0002 |

The acceptance limits were not relaxed: at least 1.05× median construction speedup, no paired
collection recall loss greater than 0.005, plus identical serial graph hashes and recall. The small
positive collection differences are observations, not a claim that an allocation optimization
improves the ANN algorithm; collection seeds and concurrent insertion schedules vary.

The parallel builder and WorkContext bytecode are identical to the baseline, and its original
uncached neighbor-selector implementation is unchanged. [Bytecode hashes](evidence/default-v7-parallel-bytecode.json)
record that check. Parallel-builder edits failed earlier recall screens and were removed entirely.
This does not turn their failed measurements into passing results.

## Allocation and regression evidence

The same isolated allocation probe fails against the baseline and passes against the candidate.
At dimension 512, 20,000 cosine calls allocate **640,000 bytes before and zero after**. It uses an
uninstrumented JVM with synchronous compilation to exclude coverage probes and partially compiled
Vector API boxes. An independent copy of the original arithmetic checks result bits across vector
boundaries and scalar tails under ordinary coverage instrumentation.

Tests also cover exact cache collisions and pruning, equal-score heap ordering, shared buffers in
the serial builder, duplicate-heavy quantizer training, replay source preservation and reopen.
Historical red/green logs are retained. Some also mention subsequently rejected code; the final
source and [PR #86 CI](https://github.com/integrallis/vectors/pull/86/checks) define the delivered scope.

## Hardware, data, and limits

Intel Core i7-9750H, 6 physical / 12 logical CPUs, 32 GiB RAM, macOS, Temurin 25.0.3, default Panama
AVX2 256-bit kernels. JVM flags: `--add-modules=jdk.incubator.vector`,
`--enable-native-access=ALL-UNNAMED`, `-Xmx6g`. No `-Dvectors.*` overrides.

Both workloads use M=16 and efConstruction=200. The collection replay preserves the pinned
DBpedia generation's COSINE metric and normalization flag, IDs, text, metadata and content hashes.
It measures indexing existing embeddings and commits, not embedding-service latency. It reads a
100,000-row prefix of the 630,000-row source and commits every 25,000 rows.

These measurements establish improvements on this host and these prefixes. They do not establish
million-row throughput, final ARM performance, or a state-of-the-art result. CI separately checks
correctness on ARM64, x86 128-bit and x86 scalar configurations.

[Reproduction commands](README.md#reproduce), [input hashes](evidence/inputs.json), and
[all timing, recall, and graph-hash values](evidence/measurements.json) accompany the individual raw
`default-v7-*` logs in `evidence/`. The complete local archive also retains rejected experiments,
source snapshots, and the original working-tree backup. All rented campaign servers were deleted.
