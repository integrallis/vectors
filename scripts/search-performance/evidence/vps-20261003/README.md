# Dedicated VPS qualification, 2026-10-03

Baseline runtime: `fc7c525e95376ceb20729e4f0bb4364b4e25a82f` (merged main).
Candidate runtime: `a29ffb86c35023a8a2d7720a7fd98681e248253a`.
Subsequent commits add benchmark evidence and correct a recovery-test assumption;
runtime source is unchanged.

Host: dedicated Hetzner ccx33, AMD EPYC Milan, 8 logical CPUs / 4 physical cores,
32 GB RAM, Ubuntu 24.04, Temurin 25.0.3+9, AVX2. Every timed JVM was pinned to
CPU 2,3 (the two hardware threads of one physical core). Default vectors runtime
settings. Three alternating fresh JVM pairs; seven samples per workload per JVM;
median within each fork and then across forks. `commands.json` records exact
commands, heap settings and classpaths. No builds/tests ran on this host during
timing. Initial input transfer overlapped the beginning of the first synthetic
warmup; later forks and the collection/published campaigns followed completed
transfers. The separate shared VPS used for recovery diagnosis supplied no timings.

`provenance.json` describes where the **jars were built**, not the measured host.
`host.txt` describes the measured Linux host. `jar-sha256.json` identifies every
frozen jar; `published-inputs.json` identifies source datasets and selected rows.
Raw logs and per-fork medians are retained for all three campaigns.

## Results

Percentages below mean **less elapsed time**, compared with main, not throughput
increases. These are scoped warm-cache search results, not SOTA, build-time or
embedding-model results. Small differences are not established gains.

| Workload | Less time |
|---|---:|
| Synthetic cosine heap batch, d128 / d512 / d768 | 21.77% / 22.82% / 29.17% |
| Synthetic cosine mapped, d128 / d512 / d768 | 7.32% / 11.41% / 9.62% |
| Saved 100k x 512 collection cosine batch / mapped | 18.22% / 10.08% |
| GloVe 100k x 100 cosine batch / mapped | 23.45% / 26.05% |
| Synthetic HNSW, ef32 / ef128 / ef512 | 7.76% / 1.81% / 6.02% |
| Saved collection HNSW, ef32 / ef128 / ef512 | 2.53% / -0.24% / 1.62% |
| GloVe HNSW, ef32 / ef128 / ef512 | 0.68% / 1.46% / 5.15% |
| Fashion-MNIST HNSW, ef32 / ef128 / ef512 | -2.18% / -0.34% / 5.35% |

The complete tables include controls and slower cases. Synthetic heap single-query
results ranged from -0.78% to +1.28%; Fashion-MNIST heap single/batch were -1.28%
and -1.13%. These paths have no new distance kernel. Filtered synthetic HNSW
improved 1.09–2.59%. No claim that every workload became faster.

All **37** workload ID/score-bit digests matched across arms and repetitions.
GloVe recall@10 (20k-row graph, 256 official test queries) was 0.83125000,
0.96484375 and 0.99960938 at ef32/128/512 in **both** revisions. Fashion-MNIST
was 0.99882813, 1.00000000 and 1.00000000 in **both** revisions. The flat corpora
were 100k GloVe and 60k Fashion-MNIST rows; the HNSW graphs used 20k-row prefixes.
Each baseline-created graph and exact oracle was reused by every candidate fork.
Searcher allocation at 20k nodes: 246,016 -> 85,976 bytes (65.05% less).

## Recovery-test failure investigated separately

Full CI on `a29ffb8` failed one repetition of the new persistence test: it assumed
that a concurrent HNSW graph always returns a stored vector as its own top hit at
a full-size beam. A stress probe reproduced a miss **on the first 200-row commit,
before failure injection or retry**. That graph contained two nodes unreachable
through any hierarchy-respecting path. Opening the exact saved generation with
main and candidate returned the same missed result; see `frozen-*.log`.

The test now verifies every stored vector exactly, returned document/score
alignment, writer-cache invalidation and identical result IDs after reopen.
It does not assert perfect ANN recall. The recall threshold and public-dataset
parity checks are unchanged. This change does not repair the pre-existing
concurrent-construction reachability issue or claim to do so. The saved diagnostic
generation remains in the workspace evidence archive.

The noisy laptop latency measurements were withdrawn and do not qualify this PR.
