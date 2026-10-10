# Changelog

All notable changes to java-vectors are documented here.

## [Unreleased]

## [0.1.29] - 2026-10-10

### Fixed

- Async ingestion completion callbacks can close the ingestor without joining their own worker.
- Scatter/gather returns after its deadline even when a client ignores interruption. Clients remain
  responsible for transport deadlines; cancellation cannot forcibly terminate an uncooperative client.
- Remove an unused demo helper and accidentally tracked Python bytecode.

### Build and dependencies

- Verify resolved internal release versions, dependency direction, and shared external policy in
  every published module. Embed the dependency contract in released jars for downstream validation.
- Align Jackson through BOM 2.22.3 (annotations 2.22) and SLF4J 2.0.20. Update the optional S3 runtime
  to AWS SDK 2.55.14, Netty 4.1.139.Final and HttpClient 5.6.5 / HttpCore 5.4.4.
  Update Commons Codec to 1.22.1, Caffeine to 3.3.0, and Avaje JSON to 3.15.
- Check published dependency graphs using separate Maven and Gradle consumers.

## [0.1.28] - 2026-10-06

### Fixed

- The GGUF quantized batched GEMV required the activation vector's length to equal `cols` exactly,
  while `out` was already allowed to be oversized and the q8 scratch arrays are documented as
  holding "at least `cols`" entries. The kernels quantize and read exactly `cols` values and never
  consult `query.length`, so the activation is now accepted as a prefix too. A model whose
  feed-forward width varies per layer reuses one buffer sized for its widest layer; requiring exact
  equality made its narrower layers fail on the quantized path while the float path, which takes
  `cols` the same way, worked. A buffer shorter than `cols` is still rejected.

- `ScalarQuantizer.decode` on an int8 quantizer sized its result by the encoded buffer's length
  rather than by the quantizer's dimension. One int8 byte holds one dimension, so a correctly sized
  buffer was always decoded correctly; a longer buffer — a pooled or reused array, or one sized for
  a wider quantizer — returned a longer vector whose tail was decoded padding. The int4 path already
  stopped at the declared dimension, and `encode` already rejects a vector that is not that long, so
  both sides of the codec now hold the same contract.

## [0.1.27] - 2026-10-04

### Performance

- Persisted full-precision HNSW COSINE queries avoid per-neighbor segment-view allocations
  and prepare the batch query norm once. On the recorded 8-vCPU AMD EPYC VPS, public queries
  at ef=100 took 10.71% less time on GLOVE and 18.21% less on DBpedia versus the preceding
  merged main. Graphs, search budgets and precision were unchanged, and all measured result-ID
  and raw score-bit digests matched. See the [query results](scripts/query-performance/RESULTS.md)
  for every workload, control, baseline and reproduction command.
- Large isolated in-memory COSINE flat queries automatically use available CPU workers, with
  a shared expansion limit and coordination for batches and concurrent bursts. The same VPS
  comparison measured 87.91% and 68.70% less single-query time on GLOVE and DBpedia respectively.
  This uses more cores and a temporary score buffer; it is not a claim of reduced CPU work.
  Euclidean and DOT scans retain their original serial scoring.
- Compatible exact COSINE batches reuse stored-vector norms and coordinate parallel scans.
  Concurrent HNSW construction and serial construction avoid additional allocation and scoring
  work. The [batch/construction results](scripts/exact-reuse/RESULTS.md) and
  [default-construction results](scripts/default-construction/RESULTS.md) retain the separate
  baselines, raw evidence and unchanged recall gates; their percentages are not additive.
- Persistent commits append eligible vector payloads instead of materializing collection-sized
  vector arrays, combine prefix checksums, stream metadata, and reuse writer graph resources.
  Prefix-aware verification and recovery handle interrupted appends. Generation subscribers,
  including object-store shipping, retain independent payload files.
- Includes the merged model-kernel work for banded GEMM and SIMD K-quant dequantization.

### Fixed

- Studio projection requests and event streams retain ownership of the latest request, so stale
  responses cannot replace its result or change a completed status to `stream closed`.
- Projection SSE endpoints replay completion, failure and cancellation to late subscribers,
  including jobs that finish while the connection is subscribing.
- Studio t-SNE honors the requested seed and allows normalized embeddings to finish the requested
  optimization budget instead of stopping at a collapsed early-exaggeration state. Non-finite or
  collapsed output produces an explicit error. Completing that budget can take longer than the
  previous premature exit; this is a correctness fix.

### Validation and compatibility

- Regression coverage includes exact score bits, filtered/tied/deleted results, query mutation,
  concurrent scan scheduling, commit recovery, projection races and t-SNE reproducibility.
  CI exercises ARM64, x86 128-bit vectors and scalar fallback, plus the unchanged recall gate.
- These optimizations apply automatically. No new runtime switch or on-disk format migration is
  required. Measured performance is specific to the retained workloads and hardware; it does not
  establish a ranking against other libraries or cold-start performance.

## [0.1.26] - 2026-10-02

Committing often no longer costs more than committing once. Every commit rebuilt the HNSW graph from
every live vector, so an ingest split into K batches paid K full builds of growing size — quadratic in
the number of commits. Measured on 8 dedicated cores, 50,000 vectors at 512 dimensions, build
throughput fell from 2,060 vec/s at a single commit to 107 vec/s when committing every 1,250, halving
each time the commit count doubled. A commit now extends the previous generation's graph and inserts
only the new vectors, which holds throughput at 1,199 vec/s at that cadence — 11.2x — with recall
unchanged.

Two correctness defects were found while validating that, both of which degraded results silently
rather than failing:

- An appended node that drew a level above the existing entry became the graph's entry point without
  being inserted, leaving it with no edges and stranding every search that started there. recall@10 on
  a persistent collection committed in six batches was 0.170.
- `graph.bin` stores neighbour ids without scores and synthesises them on decode, which is sound while
  a committed graph is read-only. Extending a decoded graph ordered its neighbours by those synthetic
  values: 568 of 1,000 vectors failed to retrieve themselves as their own nearest neighbour. Carried
  edges are now rescored from the vectors. **`HnswGraphMerger` had the same defect**, so persistent
  compaction degraded graphs in every release that shipped it.

Collections written by earlier releases are unaffected on disk and need no migration; the format is
unchanged.

## [0.1.25] - 2026-09-27

Fixes the upgrade path 0.1.24 shipped without. 0.1.24 made the manifest version an exact match, so a
0.1.23 collection stops the build — that part was intended. What went out with it was an error message
blaming the filesystem and the advice to rebuild the collection, meaning re-run an embedding model
over the whole corpus. **That advice was wrong**, and this release replaces it.

### Added

- `ManifestMigration` upgrades a collection's manifests in place, without re-embedding anything.
  Version 5 added a recipe flag at offset 160 and a 32-byte hash after it, moving the self CRC to 196
  and the header from 164 to 200 bytes. Every field version 4 carried keeps its offset, **no payload
  file changed format** — `VERSION_METADATA`, `_IDMAP`, `_QUANTIZED`, `_GRAPH` and `_TOMBSTONES` are
  all still 1 — and the content hash version 5 introduced is derived from stored text on read rather
  than persisted. So the upgrade is a few hundred bytes per generation.

  Measured on the bundled 1,929-prompt router index, a real version 4 collection: after a
  manifest-only rewrite its held-out evaluation returned **per-item results identical to a full
  rebuild** across all 481 prompts, 0.9044 either way.

  `inspect` reports, `migrate` writes, `migrationAvailable` answers the one question a caller usually
  has. The batch `main` reports by default and needs `--apply` to write, and exits 1 when a dry run
  finds work so a deployment can gate on it.

- `VectorCollectionBuilder.migrateOlderFormats(boolean)`, **off by default**. Rewriting a file inside
  somebody's collection because they happened to open it is not a default. The previous manifest is
  kept as `manifest.bin.v<n>.bak`, the rewrite is tmp/fsync/`ATOMIC_MOVE`/dir-fsync, and re-running is
  a no-op.

- `java-vectors.migrate-on-startup` in the Spring Boot starter, **off by default** — the idiom Flyway
  established, for the common case of one application owning one collection. Only applied alongside
  `storage-path`: an in-memory collection has no manifest.

- `vectors-spring-batch` (**new published module**) — `CollectionRootItemReader`,
  `ManifestMigrationItemProcessor`, `MigrationReport` and `ManifestMigrationTasklet`. Chunk-oriented
  so a bad collection can be skipped and the rest carry on; `IOException` propagates rather than being
  swallowed, because Spring Batch already owns that decision and a caught one would report a clean job
  over a collection that never migrated.

- `vectors-jakarta-batch` (**new published module**) — a JSR-352 batchlet for containers that are not
  Spring: JBeret under WildFly or Quarkus, and Open Liberty. Exit status is `MIGRATED`, `PENDING`,
  `CLEAN` or `ATTENTION`, kept apart so a JSL can branch on them; an unrecognised `apply` property is
  refused rather than read as false, since silently dry-running an operator who typed `apply="yes"`
  leaves them believing the collections were upgraded.

- Studio shows each collection's on-disk manifest format and how it relates to what this build reads,
  as `CollectionFormat` with `CURRENT`, `OLDER`, `NEWER` or `UNKNOWN`. `OLDER` is styled as a warning
  rather than an error: the collection is intact and one migration away, but every action on it fails
  until then, so it must not look ordinary.

### Fixed

- **An older collection did not appear in Studio at all.** The scan called `Manifest.readFrom`, which
  validates the version before returning, so a version 4 collection threw, was caught, and left one
  line in a log. From the UI it was indistinguishable from a collection that did not exist — the worst
  available reading, since the data is intact.

- A collection from an older format now says so instead of blaming the filesystem. Recovery used to
  bootstrap over the unreadable generation and fail with `generation directory already exists`, which
  points at directory state rather than at the format. Merged before this release but **after the
  0.1.24 tag**, so 0.1.24 shipped without it.

- The manifest header specification in `Manifest`'s javadoc still said version 4 and a 164-byte
  header. Its rows were correct; the two summary values were a version behind, in the one document
  someone reads to write a reader or migrate a collection.

### Changed

- The error for an older collection now names the migration instead of telling the reader to rebuild.
  The 0.1.24 notes below say "there is no in-place migration" and that the error "names both
  versions"; neither was true of what shipped, and both are corrected here.

## [0.1.24] - 2026-09-27

### Changed — BREAKING (on-disk format)

- **`VERSION_MANIFEST` 4 → 5. Collections persisted by 0.1.23 or earlier will not open.** The manifest
  header grows from 164 to 200 bytes to carry an embedding-recipe anchor, and the version check is
  exact rather than a floor. The error names both versions and says to rebuild. Accepted because the
  library is newly released with no known production deployments; there is no in-place migration.

### Added

- **Embedding provenance.** A collection can record *how* its vectors were produced and therefore tell
  which of them are stale. Previously a collection's only tie to the model that filled it was the
  dimension — and dimension is not identity: two unrelated 768-dimension models produce mutually
  meaningless vectors, and the store mixed them silently, so a query embedded with one and searched
  against the other returned confidently ranked nonsense with no error.
  - `EmbeddingRecipe` records model id, version and optional content digest, **separate document and
    query prefixes** (instruction-tuned embedders are asymmetric), pooling, normalisation, truncation
    policy and `maxInputTokens`, with a stable `recipeHash()` over a canonical field rendering.
  - `attestation()` distinguishes `ATTESTED` (a weights digest is recorded, so a vector can be
    re-derived and compared) from `DECLARED` (a name only, so a model swapped behind it is
    undetectable). The two are never conflated.
  - `ContentHash` — SHA-256 of the input a vector was produced from, hashed **after** chunking, prefix
    and truncation, so it answers "would re-embedding change this vector".
  - `Document.contentHash` is derived from `text` when present. It is required only where a collection
    declares a recipe: declaring *how* vectors are made creates the duty to record *what* each was made
    from. Plain vector storage is unaffected, and `Document.of(id, vector)` still works.
  - `VectorCollectionConfig.withRecipe(...)` refuses a recipe whose dimension or metric contradicts the
    collection, because a recipe that could not have produced these vectors is worse than none.
  - Recipes are **immutable once a collection holds vectors**: changing one invalidates every stored
    vector, so it is a migration rather than an update. Reopening without restating the recipe keeps it.
- **`RecipeCodec` SPI** with a dependency-free default, so the core library carries no JSON dependency.
  Discovery order: explicit argument, then `-Dvectors.recipeCodec`, then a single `ServiceLoader`
  provider, then the built-in. Two providers and no property set is an error rather than a coin toss.
  Swapping codecs is safe: `recipeHash()` comes from the canonical rendering, not the serialised text,
  so a collection written by one codec verifies under another.
- **New published module `vectors-db-jackson`** — an opt-in Jackson `RecipeCodec`.
- **Studio surfaces provenance** on the collection page and in the collections list: attested, declared
  and unknown are visually distinct, tooltips state consequences rather than states, and a collection
  with no recipe carries a written note that its rankings cannot be audited. `UNKNOWN` is deliberately
  not styled as an error — plain vector storage is a legitimate choice.

### Fixed

- **Spring AI adapter applied no instruction prefixes.** Documents and queries were both embedded bare,
  so for an instruction-tuned model such as Nomic (`search_document:` / `search_query:`) every ingest
  and every search was subtly wrong — a silent retrieval-quality loss rather than an error. The adapter
  now applies the collection recipe's document prefix at ingest and query prefix at search, hashes the
  prefixed text, and stores the original text so re-embedding does not double-apply.
- **LangChain4j adapter** now verifies an incoming embedding's dimension against the collection's
  recipe, naming that recipe's model. It cannot apply prefixes — LangChain4j embeds before calling the
  store, so no model is in scope — and that limitation is documented rather than left implied.

### Notes

- ARM **correctness** for the pinned reductions is evidenced by the aarch64 CI job. ARM **throughput**
  is unmeasured.


## [0.1.23] - 2026-09-25

### Fixed

- Make the desktop multimodal RAG distribution launch with its bundled JavaFX dependencies
  and restrict macOS Dock JVM options to macOS.
- Qualify Studio with the native ARPACK dependency needed by connected-graph UMAP projections,
  and preserve hidden UI controls when layout styles apply.
- Update Jackson and Netty in the published optional runtime dependencies to patched versions.
- Ensure Maven consumers resolve those patched runtime versions as well as Gradle consumers;
  validate each optional runtime's staged POM independently in CI and release validation.
- Run S3 and Studio integration tests in CI and release validation. Pin the S3 test emulator
  to LocalStack 4.1.0, which implements the `If-Match` conditional-write contract under test;
  the former 3.8 image silently accepted stale ETags.
- Corrected the security policy to describe the published release line and the optional S3
  runtime inside the CPU publication scope.

- Q6_K's scalar and Vector API routes now produce bit-identical results. The scalar route kept
  eight float lane accumulators, documented as existing so its reduction order would match the
  eight-lane order of the Vector API route. That was true of an older SIMD implementation; the
  current one reduces each super-block to a single integer and applies one fused multiply per
  block. The two therefore folded in different orders and disagreed by one to two units in the last
  place at every width, including a single super-block.

  **The Vector API route is unchanged, so nothing computed on a host with 256-bit vectors or wider
  moves.** The scalar route, which serves narrower hosts and the explicit scalar provider, now
  agrees with it instead of differing in the low bits. Results that previously depended on which
  route ran are now the same either way.

  The existing scalar-versus-Vector-API comparison for Q6_K asserted `offset(1e-3f)`, wide enough
  to hide the defect for as long as it existed; it now asserts bit-equality, and a Q4_K control
  runs beside it so a future failure cannot be blamed on the harness.

## [0.1.22] - 2026-09-16

### Changed

- The persistent GGUF row executor polls at every stage barrier before it parks
  (`vectors.gguf.pollMillis`, default 5 ms). The Models pure-Java backend, which issues ~200
  barriers per decode token, measured +45% decode on dedicated cores (6.4 to 9.4 tok/s) and up to
  +80% on shared vCPUs; prefill is unchanged.

### Added

- `VectorUtil.ggufPollMillis()` / `setGgufPollMillis(long)` set the barrier budget at run time for
  every persistent executor in the JVM. A caller that runs its own compute pool beside the
  executor should set 0: with both polling, prefill on the Models native backend fell from 126 to
  40 tok/s.

## [0.1.21] - 2026-09-14

### Added

- Added a little-endian `MemorySegment` factory for owned F32 execution matrices, allowing
  in-process runtimes to copy validated serialized weights into a stable Java execution layout.

### Fixed

- Made owned F32 execution matrices bit-identical before and after JIT compilation by using a
  fixed SIMD reduction tree.
- Optimized single-input F32 projections with four-row execution, improving the combined low-rank
  adapter projection shapes by 25.49% in the controlled benchmark while retaining determinism.

## [0.1.20] - 2026-09-05

### Added

- Added owned F32 execution matrices for inference runtimes that deliberately widen compact
  serialized weights once to avoid decoding them on every projection.
- Added a GGUF Q8_0-to-F32 prepared matrix with batched multiplication and explicit serialized
  versus execution-memory accounting.

### Fixed

- Added a regression gate proving Q8_0 batched matrix multiplication cannot overflow a signed
  16-bit lane reduction at the maximum block dot product.

## [0.1.19] - 2026-09-02

### Added

- Added scalar and Panama Vector API matrix-vector kernels for signed packed INT4 weights with
  FP16 group scales, including row-major and input-major layouts, batched execution, and reusable
  signed-INT8 activations.
- Added deterministic correctness tests and a retained MobileMoE-shaped JMH experiment comparing
  direct packed INT4 with rejected BF16 expansion and the selected prepared-Q8 runtime layout.

## [0.1.18] - 2026-08-30

### Added

- Added scalar and Panama Vector API kernels for SwiGLU, Qwen 3.5 gate transforms, and fused
  causal depthwise convolution with SiLU. The fused convolution preserves exact recurrent history
  semantics while removing the scalar channel loop from Qwen 3.5 inference.

## [0.1.17] - 2026-08-30

### Changed

- Reduced MXFP4 hot-loop scale access overhead without expanding mapped weights. Three-fork EPYC
  measurements improved the official GPT-OSS gate/up and down projection shapes by 5.66% and
  5.59%, while exact scalar/SIMD parity and the official-checkpoint oracle gate remained green.

## [0.1.16] - 2026-08-30

### Added

- `Mxfp4Matrix` provides zero-copy standard E2M1/E8M0 matrix views, exact F32-activation
  multiplication, and a Java Vector API W4A8 kernel with reusable prepared Q8_0 activations.
- Added deterministic encoding, approximation, activation-reuse, SIMD, and scalar coverage plus a
  reproducible JMH gate derived from the pinned FreeToken implementation.

## [0.1.15] - 2026-08-30

### Changed

- Q5_K batched matrix multiplication now reuses each unpacked weight row across a two-query
  remainder. The controlled batch-2 kernel gate improved by 40.6% with exact outputs; full groups
  of four and unsupported platform envelopes retain their established paths.

## [0.1.14] - 2026-08-29

### Changed

- Parallelized large CQ2, CQ3, CQ4, and ternary rotated-codebook row projections while retaining
  the serial path for small routing matrices. Needle-sized CQ2 and CQ4 projections improved by
  roughly `4.9x` and `5.7x`, respectively, in the targeted Java benchmark.

## [0.1.13] - 2026-08-27

### Added

- `RotatedCodebookMatrix.accepts(...)` lets Java inference runtimes safely reuse one
  prepared activation across compatible compact matrices without relying on exceptions.

## [0.1.12] - 2026-08-25

### Added

- `BFloat16Matrix` executes mapped little-endian BF16 matrix/vector and batched
  multiplication without expanding the stored weights to F32.
- `RotatedCodebookMatrix` executes CQ2, CQ3, CQ4, and ternary Hadamard-rotated
  codebook matrices, including reusable prepared activations and row slices.

### Changed

- Avaje and Jackson VCR serializers now share one canonical cassette-tree codec in
  `vectors-vcr-core`. Their JSON remains cross-compatible while new structured and streaming
  fields have a single mapping implementation.
- Mapped BF16 batches parallelize across independent rows at the provider boundary while
  preserving deterministic per-row accumulation.

## [0.1.11] - 2026-08-23

### Added

- Spring AI and LangChain4j streaming chat interfaces now record and replay ordered
  response chunks, thinking updates, and tool-call events. Dual-mode LangChain4j
  models retain both their blocking and streaming interfaces when wrapped.
- Cassettes now carry canonical SHA-256 request signatures covering model inputs,
  generation settings, tools, response formats, and framework-exposed defaults.

### Fixed

- `PLAYBACK_OR_RECORD` now replays only matching signed requests and automatically
  replaces missing, unsigned legacy, or stale interactions. Strict `PLAYBACK`
  rejects stale cassettes instead of returning a response recorded for different
  inputs or settings.
- Spring AI chat playback now reconstructs structured assistant messages, tool calls,
  all generations, generation metadata, response attributes, token usage, rate
  limits, and prompt-filter metadata rather than returning assistant text alone.
- Avaje and Jackson cassette serializers now preserve the expanded structured and
  streaming payloads while remaining interoperable with each other.

## [0.1.10] - 2026-08-23

### Fixed

- VCR model wrappers now register every successfully written cassette with the active
  JUnit 5 or TestNG test lifecycle. If a recording test fails after a model call, the
  framework deletes the cassettes written by that test instead of leaving partial
  fixtures behind.

## [0.1.9] - 2026-08-20

### Fixed

- Spring AI semantic-cache hits now preserve the response model and expose
  `vectors.semantic-cache.hit` plus the matching similarity in response metadata.
  `SemanticCachingChatModel.isCacheHit(...)` and `cacheSimilarity(...)` provide
  typed access so applications can report avoided model calls accurately.
- RAG applications can mark the user intent with
  `vectors.semantic-cache.key` instead of embedding repeated instructions and
  retrieved context as the cache key.

## [0.1.8] - 2026-08-20

### Fixed

- The Spring Boot starter now supplies the application's `ObservationRegistry`
  to `JavaVectorsVectorStore`, enabling the standard
  `db.vector.client.operation` metrics and traces when Actuator is present.

## [0.1.7] - 2026-08-07

### Added

- **Quantized-only persistent collections.** A persistent `FLAT` collection can now
  store only its compressed codes by setting `.quantizedOnly(true)`. This removes the
  full-precision `vectors.bin` payload for substantially smaller distributable indexes;
  search scores are approximate and the collection is sealed after its first commit.
  Existing collections reopen from their manifest without repeating the quantizer or
  quantized-only builder settings, and compressed payload dimensions are validated on
  open.

### Changed

- **Faster Q4 batched dot products.** The Panama kernel accumulates four Q4 blocks per
  scratch write, reducing intermediate traffic. Controlled benchmarks improved batch
  sizes 4 through 32 by roughly 4–7 percent.

## [0.1.6] - 2026-08-07

### Added

- **`vectors-router`** — semantic routing. Classifies a query into named routes by
  embedding similarity: each route carries reference phrases, and a query takes the
  route of its nearest reference when that reference is within the route's distance
  threshold. References are L2-normalized once at construction so routing is a dot
  product per candidate.

- **`vectors-cache-semantic-spring-ai`** and **`vectors-cache-semantic-langchain4j`** —
  chat-model decorators that answer from a *semantically* similar earlier prompt.
  The existing `vectors-cache-spring-ai` / `vectors-cache-langchain4j` decorators key
  on an exact request string, so "how do I reset my password" and "I forgot my
  password" are separate entries; these embed the request and serve a hit when an
  earlier one is within the cache's threshold. Responses carrying tool calls are
  returned but never stored, since replaying one would skip the tool.

- **Entry attributes and `CacheFilter` on `SemanticCache`.** Similarity alone cannot
  decide that a cached answer may be served: a completion generated at temperature
  0.9, for another tenant, or by another model can be an excellent semantic match and
  still be the wrong thing to return. Entries now carry attributes describing the
  conditions they were produced under, and `lookup(float[], CacheFilter)` scopes a
  search to the conditions the caller accepts.

  Filtering runs inside the search rather than over the nearest result. The nearest
  entry overall is frequently not the nearest *usable* one — a verbatim repeat of the
  prompt at the wrong temperature sits closer than a paraphrase at the right one — so
  testing a top-1 result after the fact reports a miss where a usable entry exists.

- `SemanticCache.lookupTopK(float[], int)` exposing the same ranking directly.

- `SemanticCache.supportsAttributes()` reporting whether an implementation stores
  attributes. An implementation that does not rejects a non-empty attribute map from
  `put` rather than dropping it, because a filtered lookup over entries with missing
  attributes serves exactly the answers the filter was added to prevent. The framework
  decorators require attribute support and check it at construction.

### Fixed

- **A Spring AI chat prompt carrying options never hit the semantic cache.** The
  decorator derived its options signature from `ChatOptions.toString()`, and
  `DefaultChatOptions` inherits the identity `toString()`, so two separately built but
  identical option sets produced different signatures and every such request missed.
  The signature is now built from the option values.

### Changed

- `SemanticCache.Hit` and `SemanticCache.Entry` gained an `attributes` component.
  Both keep their previous constructors, which default to no attributes.

## [0.1.5] - 2026-07-31

### Added

- MFCQI quality watermarks enforced in CI.

### Changed

- **Register-tiled prefill kernels for Q4_K, Q5_K and Q6_K.** Prefill now tiles across
  four queries at a time, reusing each dequantized weight block across the tile instead
  of re-reading it per query.
- Simplified MMR candidate selection in `vectors-hybrid`.

### Fixed

- Spring AI 1.0 embedding options are accepted again.

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
