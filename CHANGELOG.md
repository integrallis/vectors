# Changelog

All notable changes to java-vectors are documented here.

## [Unreleased]

## [0.1.4] - 2026-07-29

### Fixed

- **Spring Boot starter now wires the `VectorStore` regardless of auto-configuration
  order.** The auto-config resolved the Spring AI `EmbeddingModel` through
  bean-presence conditions (`@ConditionalOnBean` / `@ConditionalOnMissingBean`),
  which are evaluated while our auto-configuration is processed — before a Spring
  AI provider auto-configuration (OpenAI, etc.) had registered the model. In a
  real app this left "no VectorStore bean" and a spurious "java-vectors.dimension
  must be set" error even though an `EmbeddingModel` was present. The model is now
  resolved through an `ObjectProvider` at bean-instantiation time and the plain vs.
  Spring-AI collection beans are separated on a classpath condition, so wiring and
  dimension inference no longer depend on auto-configuration ordering. Verified
  against a real Spring Boot 4.1.0 / Spring AI 2.0.0 application.

## [0.1.3] - 2026-07-29

### Changed

- **Spring AI `JavaVectorsVectorStore` now embeds documents in batches.** `add()`
  routes the whole document list through the embedding model's batching API
  (`embed(List, EmbeddingOptions, BatchingStrategy)`) instead of one call per
  document, so ingesting N chunks costs a handful of token-limited provider
  round-trips rather than N. This also composes with the batch de-duplication in
  `CachingEmbeddingModel` and honors the model's token limit.

### Added

- `JavaVectorsVectorStore.Builder.batchingStrategy(...)` to override the batching
  strategy (defaults to `TokenCountBatchingStrategy`). The Spring Boot starter
  reuses the application's `BatchingStrategy` bean when one is present.

## [0.1.2] - 2026-07-29

### Added

- Spring Boot starter zero-config path: `java-vectors.metric` now defaults to
  `COSINE`, and the collection `dimension` is inferred from the Spring AI
  `EmbeddingModel` when unset — adding the starter with an `EmbeddingModel` bean
  is enough to get an indexed `VectorStore`. A missing dimension with no
  `EmbeddingModel` now fails at startup with an actionable message.
- `java-vectors.commit-after-add` starter property to control whether the
  auto-configured `JavaVectorsVectorStore` commits after each `add(...)`.

### Fixed

- **WAL recovery no longer bricks on a torn trailing frame.** A crash mid-append
  could leave a truncated final frame that made the segmented write-ahead log
  un-reopenable; recovery now treats an incomplete trailing frame as end-of-log
  while still surfacing a CRC mismatch on a *complete* frame as corruption.
- **Query cache is invalidated on auto-commit and compaction**, not only on the
  explicit `commit()`. Collections configured with both `autoCommitThreshold`
  and `cacheSize` no longer return stale results that omit newly added documents.
- **Query-cache keys now include the score cutoff and field projection.** Two
  otherwise-identical searches differing only in `minScore`, `includeText`, or
  `includeMetadata` no longer collide on one cache entry.
- **Spring AI embedding cache no longer aliases cached arrays.** The batched
  cache-miss path now returns a clone, so a caller mutating a returned embedding
  cannot corrupt the cached value.
- `SearchRequest` now rejects a non-positive `searchListSize` and non-positive
  `overQueryFactor`/`filterExpansion` instead of passing them through silently.

### Changed

- Product-quantizer training logs a warning when the training set has fewer
  vectors than the requested cluster count (a silent recall-collapse footgun).
- Documentation: the Spring AI guide now leads with the Spring Boot starter as
  the quick start; clarified IVF `nprobe` clamping and `VectorUtil.cosine`
  zero-norm (`NaN`) behavior.

## [0.1.1] - 2026-07-27

### Added

- `vectors-storage-s3`, an explicit opt-in artifact for the AWS SDK-backed
  `S3StorageBackend`.
- `vectors-db-arrow`, an explicit opt-in artifact for Arrow IPC import and
  export.
- The complete VCR family as Maven Central artifacts: `vectors-vcr-core`,
  `vectors-vcr-semantic-db`, `vectors-vcr-serde-avaje`,
  `vectors-vcr-serde-jackson`, `vectors-vcr-junit5`, `vectors-vcr-testng`,
  `vectors-vcr-spring-ai`, and `vectors-vcr-langchain4j`.
- A release gate that permits only Vectors modules and SLF4J in the
  `com.integrallis:vectors` runtime graph and caps the complete graph at 2 MiB.

### Fixed

- Removed unconditional AWS, Netty, Apache HTTP, Arrow, Jackson, and
  FlatBuffers transitives from the `vectors` facade. Its runtime graph is now
  9 JARs totaling 922,356 bytes, down from 63 JARs and 19,074,004 bytes.
- Release-mode notebook verification now executes all six notebooks, including
  the VCR record/replay harness, from the staged Maven artifacts.
- Empty `VECTORS_NOTEBOOK_REPOSITORY` values no longer fail source-mode
  notebook startup with an invalid URI.

## [0.1.0] - 2026-07-26

### Added

- `com.integrallis:vectors` — a single umbrella dependency that re-exports `vectors-db` (core,
  storage, quantization, and the HNSW/Vamana/IVF index backends), so applications can depend on just
  `vectors` instead of `vectors-db`.
- `Documents` — a fluent, `List<Document>` batch builder (`Documents.of("a", vec, "hello").add("b",
  vec2)`) that collapses the verbose `List.of(Document.of(...), Document.of(...))` form and drops
  straight into `VectorCollection.addAll(...)`.
- Fluent `VectorCollection.add(String id, float[] vector)` / `add(id, vector, text)` overloads that
  return the collection, so inserts chain: `collection.add("a", e1, "hello").add("b", e2).commit()`.
  Added as `default` interface methods delegating to `add(Document)` — additive and backward-compatible
  across every implementor.
- MFCQI-styled documentation site: the Antora docs get the `integrallis/mfcqi-java` look and feel
  (Space Grotesk / Inter / JetBrains Mono, monochrome palette) with a dark/light theme switch, and a
  hand-crafted marketing landing page fronts the docs on GitHub Pages (`/` landing, `/docs` Antora).
- Maven Central staging and JReleaser release automation based on the proven
  `integrallis/mfcqi-java` pipeline.
- Explicit 0.1.x publication allowlist, strict staged-artifact validation,
  CycloneDX SBOM validation, dependency locks, and an enforced 80% instruction
  coverage gate for every published module.
- Regression coverage for scalar kernels, fused similarity, core value types,
  HNSW compaction/merge, and every LangChain4j cache-store mutation overload.
- A claims-controlled JVM community launch brief and release-day demonstration
  contract.
- COSINE search performance: by default vectors are stored verbatim (a retrieved
  vector equals the original input) and scored with a fused cosine kernel. Opt into
  `VectorCollectionBuilder.normalizeCosineVectors(true)` to unit-normalize vectors at
  ingest and score them via dot product (cosine of unit vectors equals their dot
  product), collapsing the hot kernel from three reductions plus a sqrt/divide to a
  single fused dot product for a ~1-6% QPS edge on the HNSW cosine path — at the cost
  of returning normalized (not verbatim) vectors on retrieval.
- Zero-copy `MemorySegment` scoring on the persistent/mmap search path, a 4x-unrolled
  `MemorySegment` cosine kernel (parity with the `float[]` kernel), and HNSW searcher
  allocation hygiene (batched greedy descent, per-searcher reuse of scorer scratch).
- HNSW search hot-path: batch-scoring argument validation now builds its failure messages
  lazily instead of eagerly constructing `"matrix[" + i + "]"` on every row of every batch
  (which ran on the success path — ~18% of search time in a JFR profile), for a measured
  1.2-1.4x QPS speedup with recall and results bit-identical.
- HNSW search: the per-searcher visited set is now a version-stamped `int[]` tag array with an
  O(1) generation-bump reset, replacing a `java.util.BitSet` whose per-query `clear()` zeroed
  all `graph.size()` bits every search (an O(N) cost that scales with corpus size). Mirrors
  hnswlib's `visited_list_pool`; recall/results bit-identical.
- Zero-copy `MemorySegment` scoring on the **Vamana** disk/mmap search path (parity with the
  existing HNSW segment path): scores directly off the stored vector's segment slice instead of
  copying each candidate mmap→`float[]`. Adds `supportsSegments()`/`vectorSegment()` to the Vamana
  `RandomAccessVectors` interface.
- `SemanticCache.Hit` now exposes the matched entry's `key`, so callers can tell *which* cached
  entry answered a near-duplicate lookup, not just its payload.
- A configurable per-file size cap on `DirectorySource` (`maxFileBytes`, default 64 MiB — matching
  TurboPuffer's max-document limit) that fails fast on an oversized file instead of OOMing the heap.
- `PartialResultException` (carrying the partial total plus the unreachable-node set), thrown by the
  now fault-tolerant, timeout-bounded distributed `size()`/`physicalSize()`.
- Object-storage manifest primitives (`StorageManifest` + `ManifestStore`): a versioned, CAS-published
  pointer (monotonic `generation`, content hash, shard→generation map) advanced via the object-storage
  conditional-put — the object-storage-native commit-point pattern used by Iceberg/Delta/Lance/
  TurboPuffer, with an optimistic read→rebase→CAS commit loop.
- The object-storage `DistributedVectorCollection` now publishes a **durable,
  CAS-protected generation manifest** on `commit()` (and resolves it on `open()`): the committed
  generation — previously in-memory only — is now discoverable by remote/replica readers and monotonic
  across concurrent writers, with an optional gossip announce bridge (`setGenerationAnnouncer`) into
  `announceVersion`. The WAL remains the authority for local crash recovery; the manifest is the
  discoverable, race-safe object-storage pointer.
- `DistributedVectorCollection` now writes cluster payloads under **generation-scoped keys**
  (`gen-<N>/cluster-<id>`)
  and the manifest records a per-cluster generation map. A commit rewrites only the clusters it
  dirtied — advancing just those to the new generation (no write amplification) — and the manifest
  CAS is the atomic commit point that switches the live per-cluster generation set. A crash after the
  new-generation objects are written but before the manifest CAS therefore leaves the prior
  generation's objects intact and merely orphans the unreferenced new ones (content-addressed-payload
  commit, as in Iceberg/Delta/Lance); `open()` resolves each cluster's payload key from the manifest.

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
- **Ingest durability/correctness** (module audit): retry no longer duplicates a partially-staged
  batch (`DistributedVectorSink.addAll` is idempotent); the persisted resume cursor is now loaded on
  start, so a restart resumes instead of re-ingesting from offset 0; `R2KeyCursor` advances its
  in-memory offset only *after* the durable write succeeds; `JsonlSource` closes its reader on early
  abort (FD leak) and the ingest producer closes the source iterator in a `finally`; `DirectorySource`
  walks the tree once (memoized) instead of once per `estimatedSize()`/`iterator()` call.
- **Distributed/cluster availability & consistency**: `size()`/`physicalSize()` are now
  timeout-bounded and fault-tolerant (parallel fan-out, partial-with-signal on node failure) instead
  of a hang-prone sequential loop; BuoyIndex version gossip applies a monotonic max-wins guard so a
  stale/reordered announce can no longer clobber a newer index or trigger a stale reload.
- **Search-quality guards**: hybrid fusion rejects NaN scores at the source (`ScoredId`), preventing
  top-k poisoning; IVF COSINE search guards a zero stored/query norm (no NaN into the heap, no crash
  on a zero query); `SemanticRouter` rejects duplicate route names instead of silently overwriting
  exemplars.
- **Optimizer**: `GridSampler` log-scale `IntRange` enumerates distinct grid points (no duplicate
  trials / over-reported `total()`); the router/cache threshold studies now fold accuracy and
  measured latency/cost through the configured `ObjectiveWeights` instead of hardcoding the objective
  to accuracy, and `CacheThresholdStudy` scores the correct matched key against the expected label
  rather than "any hit".

### Changed

- Rewrote release-facing documentation around the implemented single-process
  CPU scope and removed unsupported performance, scale, distributed, and GPU
  claims.
- `GossipClusterMembership.announceVersion(String)` → `announceVersion(long generation, String hash)`
  to carry the monotonic generation the version guard compares on.
- The router/cache threshold optimizer studies take an `ObjectiveWeights`; the prior constructors
  delegate with a recall-only default, preserving the previous accuracy-only behaviour.
- `ClusterVectorCollection.commit()`'s javadoc now documents that it is **not** cross-shard atomic (a
  mid-fan-out failure can leave shards on different generations) — reconciling the overridden
  `VectorCollection.commit()` "atomically installs a new generation" contract.
