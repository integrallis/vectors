# Optimizer Tutorial

Hands-on walkthrough of `vectors-optimizer`. In four runnable Gradle stages you go from a default
HNSW configuration on a real ANN-Benchmarks dataset to a tuned configuration that retrieves the
same answers ~5× faster, and finish with a router-threshold sweep.

> Sample run on `fashion-mnist-784-euclidean` (10 K corpus / 200 queries):
>
> | Metric          | Baseline (M=16, efC=200) | TPE-best (M=8, efC=42, efSearch=26) | Δ        |
> | --------------- | -----------------------: | ----------------------------------: | -------: |
> | recall@10       |                   1.0000 |                              0.9940 | −0.6 %   |
> | build time      |                  5318 ms |                             1253 ms | **−4.2×** |
> | p95 latency     |                   180 µs |                                36 µs | **−5.0×** |
> | objective score |                   1.2137 |                              1.4294 | +0.22    |

## Prerequisites

- JDK 21 (the project uses virtual threads, records, and the Panama Vector API).
- ≈ 100 MB free disk under `~/.vectors/` for the cached dataset and study artefacts.
- Internet access on the **first** run only — `DatasetDownloader` fetches
  `fashion-mnist-784-euclidean.hdf5` (≈ 30 MB) and caches it locally.
- About 7 minutes of wall-clock time for stages 1–4 on a developer laptop.

No additional setup is required: the Gradle wrapper, the dataset downloader, and the optimizer's
study persistence are all bundled.

## Quickstart (TL;DR)

```sh
./gradlew :demos:optimizer-tutorial:stage1Baseline    # ≈ 10 s after dataset cache
./gradlew :demos:optimizer-tutorial:stage2BroadSweep  # ≈ 3.5 min  (20 random trials)
./gradlew :demos:optimizer-tutorial:stage3Tpe         # ≈ 3.5 min  (40 TPE trials)
./gradlew :demos:optimizer-tutorial:stage4Threshold   # ≈ 1 s       (125 grid points)
```

Each stage prints a tabular report and persists a snapshot under `~/.vectors/optimizer/`. Stage 3
reads Stage 1's snapshot and renders a side-by-side before/after comparison — that is the headline
artefact of the tutorial.

## What the tutorial teaches

On a 10 K-vector real-world corpus, HNSW with default parameters already achieves 100 % recall@10.
The tutorial is therefore not about *more* recall — it is about **finding the cheapest
configuration that preserves recall**. That is the everyday production trade-off `vectors-optimizer`
exists to automate:

- Stage 1 establishes the **baseline** — the numbers you get with zero tuning.
- Stage 2 uses **Random search** to discover *which regions* of the parameter space are promising.
- Stage 3 uses **TPE** (Tree-structured Parzen Estimator) to *exploit* those regions and converge
  on a much better configuration with the same wall-clock budget.
- Stage 4 demonstrates that the optimizer is not just for index parameters — it tunes
  **decision-boundary thresholds** on a `SemanticRouter` the same way.

## Dataset

By default the tutorial loads **`fashion-mnist-784-euclidean`** (60 K × 784, ≈ 30 MB), downloaded
on first run and cached under `~/.vectors/ann-benchmarks/`.

The corpus is **stride-subsampled** to 10 K vectors with 200 queries by default so each sweep
finishes in a few minutes. Ground truth is **recomputed against the subsample** with a brute-force
FLAT search at load time — the bundled `/neighbors` group references ordinals in the full corpus,
so re-using it would artificially cap recall.

### Overriding dataset & size

| Environment variable                  | Effect                                                                  | Default                          |
| ------------------------------------- | ----------------------------------------------------------------------- | -------------------------------- |
| `VECTORS_OPTIMIZER_TUTORIAL_DATASET`  | ANN-Benchmarks dataset name (any entry from `DatasetRegistry`)          | `fashion-mnist-784-euclidean`    |
| `VECTORS_OPTIMIZER_TUTORIAL_FULL`     | Set to `1` to use the full corpus & query set instead of subsampling    | unset (subsample 10 K / 200)     |

Useful alternates:

- `sift-128-euclidean` (≈ 500 MB, the canonical ANN benchmark; non-saturating recall — good if you
  want a story focused on recall improvement rather than latency).
- `glove-100-angular` (≈ 460 MB, angular distance, semantic-search shaped data).

```sh
VECTORS_OPTIMIZER_TUTORIAL_DATASET=sift-128-euclidean \
  ./gradlew :demos:optimizer-tutorial:stage1Baseline
```

## Stage 1 — Baseline

```sh
./gradlew :demos:optimizer-tutorial:stage1Baseline
```

Builds a `VectorCollection` with the library defaults (`M=16`, `efConstruction=200`) and runs a
single trial through `IndexStudy.runOne`. Measures `recall@10`, `NDCG@10`, and `p50/p95/p99`
latency over 1 warmup + 3 measurement rounds, then writes
`~/.vectors/optimizer/tutorial/baseline.json`.

Expected output (numbers will vary slightly with hardware):

```text
=== Stage 1 — Baseline (default HNSW: M=16, efConstruction=200) ===
Dataset: fashion-mnist-784-euclidean   corpus=10000   queries=200   dim=784

Baseline
┌──────────────────────┬─────────────────────────────────┐
│ Build time           │                       5318 ms │
│ Recall@10            │                        1.0000 │
│ NDCG@10              │                        1.0000 │
│ Latency p50 / p95    │           123.7 µs / 180.0 µs │
│ Objective score      │                        1.2137 │
└──────────────────────┴─────────────────────────────────┘
```

**How to read it.** The objective score is a weighted scalar — see
[Objective weights](#objective-weights) below. Recall@10 = 1.0 means the index returns the same
top-10 neighbours as a brute-force FLAT search; the build and latency numbers are the cost of
that perfect recall. Stage 3 will show how much of that cost is avoidable.

## Stage 2 — Broad random sweep

```sh
./gradlew :demos:optimizer-tutorial:stage2BroadSweep
```

Twenty-trial **Random search** over `m ∈ [8, 64]` and `efConstruction ∈ [40, 300]`. Trials are
appended one-per-line to `~/.vectors/optimizer/studies/tutorial-stage2-random.jsonl` via
`StudyStore` so you can `tail -f` the file while the sweep runs. The console prints the top-5
configurations by objective score:

```text
=== Stage 2 — Broad random sweep (20 trials, m & efConstruction) ===
Dataset: fashion-mnist-784-euclidean   corpus=10000   queries=200   dim=784

Top 5 trials by objective:
  #    m       efConstruction   recall@10  ndcg@10    p95 latency   score
  1    9       55               1.0000     1.0000     102.1     µs  1.3648
  2    11      44               0.9995     0.9991     103.7     µs  1.3562
  3    8       98               0.9995     0.9975     116.1     µs  1.3478
  4    11      235              1.0000     1.0000     114.3     µs  1.3158
  5    11      287              1.0000     1.0000     116.8     µs  1.3080

20 trials persisted to ~/.vectors/optimizer/studies/tutorial-stage2-random.jsonl
```

**How to read it.** The Random sampler makes no assumptions about the objective surface — it just
covers the space. Notice that the best Random trial already beats the baseline by ~43 % on p95
latency at recall = 1.0; this tells us small `m` values dominate, which is exactly the kind of
*regional* hint Stage 3's TPE sampler will exploit.

## Stage 3 — TPE optimization (the headline)

```sh
./gradlew :demos:optimizer-tutorial:stage3Tpe
```

Forty-trial **Tree-structured Parzen Estimator** study over `m`, `efConstruction`, *and* the
search-time `efSearch` axis. TPE models `p(params | score above quantile)` and
`p(params | below)` and proposes the next trial by maximising their ratio — concentrating samples
in the region Random discovered. The stage reads `baseline.json` written by Stage 1 and renders
the headline before/after table:

```text
=== Stage 3 — TPE optimization (40 trials, m & efConstruction & efSearch) ===
Dataset: fashion-mnist-784-euclidean   corpus=10000   queries=200   dim=784

Before vs. after
┌─────────────────────┬───────────────┬───────────────┬───────────────┐
│ Metric              │      Baseline │      TPE-best │             Δ │
├─────────────────────┼───────────────┼───────────────┼───────────────┤
│ recall@10           │        1.0000 │        0.9940 │ ✗     -0.0060 │
│ ndcg@10             │        1.0000 │        0.9935 │ ✗     -0.0065 │
│ p95 latency (µs)    │         180.0 │          36.3 │ ✓      -143.7 │
│ build time (ms)     │          5318 │          1253 │ ✓       -4065 │
│ objective score     │        1.2137 │        1.4294 │ ✓     +0.2157 │
└─────────────────────┴───────────────┴───────────────┴───────────────┘
Best params: m=8  efConstruction=42  efSearch=26
```

**How to read it.** ✓/✗ marks whether each metric improved relative to the baseline (higher is
better for recall/ndcg/score, lower is better for latency/build time). The 0.6 % recall
regression is intentional — the objective weights tell the optimizer it is allowed to trade a
little recall for large latency wins. If you want recall preserved exactly, raise
`recallWeight` substantially or set a hard floor in your own objective — see
[Adapting to your own data](#adapting-the-tutorial-to-your-own-data).

## Stage 4 — Router threshold sweep

```sh
./gradlew :demos:optimizer-tutorial:stage4Threshold
```

Three-route `SemanticRouter` (sports / food / weather) over a deterministic 3-D embedding space.
A `5 × 5 × 5` Grid sweeps each route's distance threshold over `{0.10, 0.20, 0.30, 0.40, 0.50}`
against a 15-probe labelled set that includes three off-topic queries (the optimum must *both*
admit in-class queries and reject off-topic ones).

```text
=== Stage 4 — Router threshold sweep (3 routes × 5-point grid = 125 trials) ===
Evaluated 125 threshold combinations in 73 ms.

Best configuration
┌─────────────────────┬──────────┐
│ threshold_sports    │ 0.10     │
│ threshold_food      │ 0.10     │
│ threshold_weather   │ 0.10     │
├─────────────────────┼──────────┤
│ accuracy            │ 0.8667   │
└─────────────────────┴──────────┘
```

**How to read it.** This stage is intentionally synthetic and fast (≈ 100 ms) — the point is to
show that threshold optimisation is a *different problem* than index optimisation. Nothing about
the corpus changes; only the per-route decision boundary at query time. The same accuracy-based
objective machinery, persistence, and Studio dashboard work identically here.

## Objective weights

Stages 1–3 share the same `ObjectiveWeights`:

```text
recallWeight          = 1.0
ndcgWeight            = 0.5
latencyP95Weight      = 0.5    (latencyP95ReferenceUs   =    500.0)
buildTimeWeight       = 0.3    (buildTimeReferenceMs    = 15_000.0)
```

The composite score is a weighted sum where minimisation metrics are mapped to
`max(0, 1 − value/reference)` so each axis lives on a comparable `[0, 1]` scale. Recall remains
the dominant signal, but latency and build time receive enough weight that the optimiser will
sacrifice a small recall margin for a large latency win — exactly the trade-off production teams
want to make explicit. Re-tuning these four numbers is the easiest way to redirect the optimiser
toward your own quality/cost target.

## Outputs

| Path                                                              | Written by | Format                       |
| ----------------------------------------------------------------- | ---------- | ---------------------------- |
| `~/.vectors/optimizer/tutorial/baseline.json`                     | Stage 1    | JSON snapshot                |
| `~/.vectors/optimizer/tutorial/best.json`                         | Stage 3    | JSON snapshot                |
| `~/.vectors/optimizer/studies/tutorial-stage2-random.jsonl`       | Stage 2    | JSON-Lines per `TrialResult` |
| `~/.vectors/optimizer/studies/tutorial-stage3-tpe.jsonl`          | Stage 3    | JSON-Lines per `TrialResult` |

The `.jsonl` files are appended one line per trial as the study runs, so a long-running sweep can
be inspected (`tail -f`) while it executes — the same format the Studio UI consumes via SSE.

## Troubleshooting

| Symptom                                                                                  | Likely cause / fix |
| ---------------------------------------------------------------------------------------- | ------------------ |
| Stage 3 prints *"No baseline snapshot found"*                                            | Run `:stage1Baseline` first; Stage 3 reads `baseline.json` from `~/.vectors/optimizer/tutorial/`. |
| First run hangs at "Downloading…"                                                        | The HDF5 download is single-threaded and ≈ 30 MB; allow up to ~30 s on a slow link. The file is cached for subsequent runs. |
| `IllegalStateException: dimension mismatch`                                              | A previous run cached a different dataset under the same collection name. Remove `~/.vectors/optimizer/tutorial/` and re-run. |
| `recall@10` is much lower than the sample (e.g. ≈ 0.1)                                   | You replaced `TutorialDataset` but did not recompute ground truth on your subsample. Mirror `TutorialDataset.computeFlatGroundTruth` — a FLAT search over the same subsample. |
| Stage 2 / Stage 3 take much longer than the quoted ≈ 3.5 min                             | `VECTORS_OPTIMIZER_TUTORIAL_FULL=1` is set, which uses the *full* corpus — for the Fashion-MNIST default that's 60 K vectors and each trial pays the proportionally larger build cost. Unset the variable to return to the 10 K subsample. |
| Re-running Stage 2 keeps appending to the same `.jsonl`                                  | By design — `StudyStore` is append-only. Delete the file under `~/.vectors/optimizer/studies/` to start a fresh study. |
| Verbose `io.jhdf` debug logs                                                             | The bundled `src/main/resources/logback.xml` should suppress them. If it doesn't, ensure no other logback config is on the classpath ahead of the demo's. |

## Adapting the tutorial to your own data

`TutorialDataset` is the only file that touches the dataset SPI — replace it with your own loader
and the four stages will work unchanged. Two things are required:

1. A `List<Document>` corpus where each document carries a stable id and a vector of constant
   dimension.
2. A `Queries` map plus matching ground-truth `Qrels`. For relevance-graded data use your own
   judgements; for pure ANN-style data, the canonical recipe is to compute top-K from a FLAT
   search on the same corpus.

The `EmbeddingProvider` returned by `embedder()` is what `IndexStudy` calls to convert query ids
back into vectors at search time; for a precomputed-vector dataset it is just a hash-map lookup,
but for a text-search dataset it would delegate to the same embedding model used at ingestion.

Once `TutorialDataset` is yours, the next places to adjust are:

- `ObjectiveWeights` (Stages 1/2/3) — match the recall/latency/build-time/memory trade-off your
  service actually cares about.
- `SearchSpace` (Stages 2/3) — narrow the ranges to your hardware envelope, or add quantizer /
  metric / index-type axes (see `vectors-optimizer/README.md` for the full SPI).
- `nTrials` and `samplerKind` — start with Random for discovery, switch to TPE once you know the
  promising region.

## See also

- [`vectors-optimizer/README.md`](../../vectors-optimizer/README.md) — full SPI reference for
  search spaces, samplers, studies, and persistence.
- The Studio UI exposes the same studies live — `./gradlew :vectors-studio-web:run` then open a
  collection and click *Optimize*.
