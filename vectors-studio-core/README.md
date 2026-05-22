# vectors-studio-core

UI-agnostic domain core for **Vectors Studio** — backend SPI, dimensionality-reduction projections, dataset analysis, and algorithm recommender. Consumed by `vectors-studio-web` and any other Studio frontend.

## System Requirements

> **Native BLAS/LAPACK (`libopenblas`) is required.** The Smile-backed projections (PCA, and the
> SVD steps inside t-SNE/UMAP initialisation) perform their linear algebra through native
> LAPACK/BLAS.

Install it before building or running anything that uses this module:

| Platform | Command |
|----------|---------|
| Debian / Ubuntu | `sudo apt-get install -y libopenblas-dev` |
| Fedora / RHEL | `sudo dnf install -y openblas-devel` |
| macOS (Homebrew) | `brew install openblas` |
| Arch | `sudo pacman -S openblas` |

The `-dev` / `-devel` package is recommended because it installs the bare `libopenblas.so`
symlink that Smile's loader looks for (the runtime-only package installs `libopenblas.so.0`).

This is the one native dependency in the Vectors stack — it is confined to Studio's projection
math. The core retrieval modules (`vectors-core`, `vectors-db`, indexes, quantization) remain
pure-Java with no JNI.

## Responsibility

- Sealed `StudioBackend` SPI with two implementations: in-process (`EmbeddedStudioBackend`, opens a directory of `vectors-db` collections) and HTTP (`RemoteStudioBackend`, layered on `VectorsServerClient`)
- `StudioBackendFactory.open(ConnectionConfig)` chooses between them
- `MetadataAdapter` — pulls per-document metadata into Studio's `DocumentView`
- Per-collection metadata schema cache (`MetadataSchema`, `FieldSpec`, `FieldType`)
- Pluggable `Projection` SPI plus Smile-backed PCA, t-SNE, and UMAP implementations
- `ProjectionRunner` — synchronous dispatch from `ProjectionRequest` to the right `Projection`
- `DatasetStats` + `DatasetStatsCollector.analyze(backend, collection, sampleSize)` — fast statistical fingerprint for the recommender
- `HeuristicRecommender` — pure function from `DatasetStats` → `ProjectionRecommendation` (PCA/t-SNE/UMAP picked from N, dim, and sparsity)
- Optional `LlmRecommender` — wraps a langchain4j `ChatModel` (compile-only) and adds a narrative `llmExplanation` to a recommendation
- `StudioSession` — per-process façade bundling backend + heuristic + optional LLM + projection runner + per-collection schemas

## Key Types

- `StudioBackend` (sealed): `listCollections()`, `describe(name)`, `search(name, SearchSpec)`, `getDocument(name, id)`, `getBlob(name, id)`, `previewDocuments(name, offset, limit)`, `documentPage(name, offset, limit)`, `vectorBatch(name, ids)`, `streamAllVectors(name, sink, progress)`, `commit(name)`, `deleteCollection(name)`, `close()`
- `EmbeddedStudioBackend.open(Path dataDir)` / `RemoteStudioBackend.open(ConnectionConfig.Remote)`
- `ConnectionConfig` (sealed): `Embedded(Path dataDir)`, `Remote(URI baseUrl, String token, Duration timeout)`
- `CollectionSummary` — name, dimension, metric, indexType, size, indexParams
- `SearchSpec`, `SearchHit`, `DocumentView` — transport-agnostic search records
- `DocumentPageView(items, total)` — atomic page + total returned by `documentPage`
- `ContentKind` — enum `TEXT / JSON / IMAGE / AUDIO / BINARY / EMPTY` with `detect(DocumentView)` for type-aware viewer dispatch
- `Projection` — `run(float[][] data, ProgressListener listener)` returning `ProjectionResult`
- `ProjectionAlgorithm` — enum `PCA`, `TSNE`, `UMAP`
- `ProjectionParams` (sealed): `PcaParams(components, center, whiten)`, `TsneParams(perplexity, learningRate, iterations, seed)`, `UmapParams(neighbors, minDist, iterations, seed)`
- `ProjectionRequest(collection, algorithm, dimensions, sampleSize, params)`, `ProjectionResult(coords, algorithm, durationMs)`
- `ProgressListener` — `onIteration(int step, int total, double loss)`
- `HeuristicRecommender.recommend(DatasetStats, int targetDims)` → `ProjectionRecommendation(algorithm, params, rationale, llmExplanation)`
- `LlmRecommender.enrich(DatasetStats, ProjectionRecommendation)` → enriched copy
- `StudioSession.backend()`, `heuristic()`, `llm()`, `projectionRunner()`, `metadataSchema(name)`, `putMetadataSchema(name, schema)`

## Dependencies

- `vectors-core`
- `vectors-db`
- `vectors-server-client`
- `com.github.haifengl:smile-core:6.0.0` — **requires native `libopenblas` at runtime** (see [System Requirements](#system-requirements))
- `dev.langchain4j:langchain4j-core:1.13.1` (compile-only)
