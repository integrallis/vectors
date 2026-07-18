# jmh-results/ — illustrative baselines, not authoritative

The `*.txt` files in this directory are snapshots of a single JMH run on a
single machine — they exist so that reviewers can see roughly what the
kernels score, and so a regression that drops throughput by 5x is visible
in a `git diff`.

They are **not authoritative**:

- Numbers are tied to a specific JDK (recorded in each file's header — JDK
  25.0.x at the time of writing), a specific JMH version (1.37), and a
  specific CPU (Apple Silicon NEON 128-bit / x86_64 AVX-512 — the file's
  header notes which). A different machine will produce different
  numbers; that does not constitute a regression.
- `@Fork(1)` keeps the suite cheap to run on a laptop. For a publication-
  grade measurement raise `@Fork` to ≥ 3 and rerun on the target hardware.
- Recall numbers may use the dataset under `~/datasets/` (see
  `DatasetRegistry`); reproducing them locally requires fetching the
  matching dataset first.

**To use these for a regression gate, regenerate them in your environment
first** — the gate is "did *my* number move", not "did it match the
checked-in baseline".

Focused JVM/compiler comparisons may also be recorded as Markdown reports. Unlike the illustrative
single-fork snapshots, each report must state its fork count, runtime revision, affinity, CPU
features, generated-code evidence, and the limitation of its conclusion. See
`vector-api-pairwise-jvm-baseline.md` for the first such comparison.

Audit T4.11 (2026-06-06) noted this nuance. The `System.gc()` calls in
the hand-rolled benchmarks have been audited; the only ones that remain
are in `BuildScalabilityBenchmark`, where they're justified for heap-size
measurement (and are commented as such). The latency benchmarks
(`RecallQpsBenchmark`, `QuantizationRecallBenchmark`) no longer call
`System.gc()` between warmup and measurement — the GC pauses they
previously suppressed are part of the latency distribution we want to
publish.
