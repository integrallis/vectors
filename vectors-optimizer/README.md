# vectors-optimizer

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-optimizer/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

Hyperparameter optimization for the `vectors-*` retrieval stack. Sweep index
parameters (HNSW / Vamana / IVF), search-time knobs, quantizer choices,
similarity metric, semantic-router thresholds, and semantic-cache thresholds
using a Java-native sampler suite (Grid / Random / univariate TPE) against a
weighted scalar of recall, NDCG, precision, F1, MRR, latency, build time, and
memory.

## Modules at a glance

| Package                       | Purpose                                                                 |
| ----------------------------- | ----------------------------------------------------------------------- |
| `space`                       | `ParamSpec` (sealed), `SearchSpace`, `Trial`, `ScoredTrial`             |
| `sampler`                     | `GridSampler`, `RandomSampler`, `TpeSampler`, `ParamSampler` SPI        |
| `data`                        | `Qrels`, `Run`, `Queries`, `MetadataQrelsDeriver`                       |
| `eval`                        | `Metrics` (Recall@k, NDCG@k, Precision@k, F1@k, MRR)                    |
| `objective`                   | `ObjectiveWeights`, `Direction`, composite `Objective` scorer           |
| `embed`                       | `EmbeddingProvider` SPI                                                 |
| `study`                       | `IndexStudy`, `RouterThresholdStudy`, `CacheThresholdStudy`, `StudyRunner` |
| `persist`                     | `StudyStore` (append-only JSON-Lines per study)                         |

## End-to-end tutorial

The [`demos:optimizer-tutorial`](../demos/optimizer-tutorial/README.md) module walks through the
full progression — baseline → broad sweep → TPE → threshold tuning — on the
`fashion-mnist-784-euclidean` ANN-Benchmarks dataset (auto-downloaded). Sample run:

```text
Before vs. after
┌─────────────────────┬───────────────┬───────────────┬───────────────┐
│ Metric              │      Baseline │      TPE-best │             Δ │
├─────────────────────┼───────────────┼───────────────┼───────────────┤
│ recall@10           │        1.0000 │        0.9940 │ ✗     -0.0060 │
│ p95 latency (µs)    │         180.0 │          36.3 │ ✓      -143.7 │
│ build time (ms)     │          5318 │          1253 │ ✓       -4065 │
│ objective score     │        1.2137 │        1.4294 │ ✓     +0.2157 │
└─────────────────────┴───────────────┴───────────────┴───────────────┘
Best params: m=8  efConstruction=42  efSearch=26
```

```sh
./gradlew :demos:optimizer-tutorial:stage1Baseline
./gradlew :demos:optimizer-tutorial:stage2BroadSweep
./gradlew :demos:optimizer-tutorial:stage3Tpe
./gradlew :demos:optimizer-tutorial:stage4Threshold
```

## Quickstart — Index study

```java
SearchSpace space = new SearchSpace(List.of(
    new ParamSpec.FixedString("metric", "COSINE"),
    new ParamSpec.FixedString("indexType", "HNSW"),
    new ParamSpec.IntRange("m", 8, 64, true),
    new ParamSpec.IntRange("efConstruction", 50, 400, true)));

ObjectiveWeights weights = ObjectiveWeights.builder()
    .recallWeight(1.0)
    .ndcgWeight(0.5)
    .latencyP95Weight(0.1)
    .latencyP95ReferenceUs(1_000)
    .build();

StudyConfig cfg = StudyConfig.builder()
    .searchSpace(space)
    .objectiveWeights(weights)
    .samplerKind(StudyConfig.SamplerKind.TPE)
    .corpusSource(() -> myCorpus)
    .qrelsSource(() -> myQrels)
    .queriesSource(() -> myQueries)
    .nTrials(40)
    .kForMetrics(10)
    .seed(42L)
    .build();

StudyStore store = StudyStore.defaultRoot();
IndexStudy study = new IndexStudy(cfg, myEmbedder);
try (StudyRunner runner = new StudyRunner("study-001", study, cfg, store)) {
  runner.runBlocking();
}

TrialResult best = store.loadAll("study-001").stream()
    .max(Comparator.comparingDouble(TrialResult::objectiveScore))
    .orElseThrow();
```

## Sampler selection

| Sampler  | Use when                                                                 |
| -------- | ------------------------------------------------------------------------ |
| `Grid`   | Small discrete spaces; you want exhaustive coverage. Continuous axes are rejected. |
| `Random` | Default baseline. Works for any axis type; supports log-scale ranges.    |
| `TPE`    | Budget ≥ 20 and trials are expensive. Univariate Parzen estimator with a configurable startup phase. |

`TpeSampler` exposes hyperparameters via `TpeSampler.Hyperparameters(nStartupTrials, nEiCandidates, gamma, priorWeight)`.
The defaults (`10, 24, 0.25, 1.0`) match Optuna's univariate TPE for parity.

## Threshold studies

`RouterThresholdStudy` accepts a `RouterFactory` that builds a fresh
`SemanticRouter` from per-route thresholds (axis names of the form
`threshold_<route>`); each trial scores micro-averaged accuracy across a
`List<LabeledQuery>`.

`CacheThresholdStudy<V>` accepts a `CacheFactory<V>` that builds a fresh
`SemanticCache<V>` from one threshold (axis name `threshold`). Seed entries
populate the cache before the labelled probes run; accuracy treats expected
hits and expected misses symmetrically.

## Studio integration

`vectors-studio-web` ships an Optimize page (linked from every collection
overview) that drives an `IndexStudy` over the active collection: it derives a
qrels set either from a metadata field via `MetadataQrelsDeriver` or, when no
field is provided, from FLAT-search self-similarity. Trials are persisted to
`~/.vectors/optimizer/studies/{studyId}.jsonl`; the page renders an
optimization-history line chart (running best) and a sortable trial table fed
by an SSE stream.

REST surface:

| Method | Path                                                       | Purpose                          |
| ------ | ---------------------------------------------------------- | -------------------------------- |
| GET    | `/collections/{name}/optimize`                             | Study-design page                |
| POST   | `/api/optimize/studies`                                    | Submit a new study (returns id)  |
| GET    | `/api/optimize/studies/{id}/events`                        | SSE event stream                 |
| GET    | `/api/optimize/studies/{id}/trials`                        | JSON list of completed trials    |
| POST   | `/api/optimize/studies/{id}/cancel`                        | Cooperative cancel               |
| POST   | `/collections/{name}/optimize/apply/{study}/{trial}`       | Capture trial params (`?confirm=true` required) |

## Persistence layout

```
~/.vectors/optimizer/studies/
  ├── {studyId}.jsonl       # one TrialResult per line, append-only
  └── {studyId}.meta.json   # optional sidecar describing study config
```

`StudyStore` synchronises writes per study with a `ReentrantLock`, so a single
JVM can run several studies in parallel without interleaving lines.
