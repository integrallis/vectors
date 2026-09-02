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
`vector-api-pairwise-jvm-baseline.md` for the first such comparison and
`mapped-kquant-jdk25-jdk26.md` for the controlled mapped GGUF K-quant gate.

`mxfp4-scale-access-epyc-20260830.json` records a three-fork A/B comparison at both
GPT-OSS 20B expert projection shapes, plus the official-checkpoint prefill screen and a rejected
heap-copy variant.

## Experiment journal

This index records the sequence of questions, evidence, and decisions behind the quantized Java
kernels. It links to the retained measurements instead of copying results into a second source of
truth. A retained result is still limited to the hardware and runtime envelope in its report.

| Date | Question | Evidence and finding | Decision |
| --- | --- | --- | --- |
| 2026-07-18 | Can the Vector API express the pairwise integer operations used by packed Q4/Q8 kernels? | [`vector-api-pairwise-jvm-baseline.md`](vector-api-pairwise-jvm-baseline.md) shows that the tested Graal fix lowers the expanded graph to `VPMADDWD` and `VPMADDUBSW`; HotSpot C2 does not. The complete Q8 block remained slower even after the desired instructions appeared. | Keep the established Q8 kernel. Retain the report as evidence that instruction selection is necessary but not sufficient, and that mixed-byte pairwise lowering plus AArch64 dot-product lowering remain JVM opportunities. |
| 2026-07-18 | Do long-offset mapped-memory reads improve K-quant execution on Java 25 or 26? | [`mapped-kquant-jdk25-jdk26.md`](mapped-kquant-jdk25-jdk26.md) records exact JMH and MiniCPM5 gates. Benefits vary by quantization and JDK; Java 25 remained faster than Java 26 for the measured Q4_K kernel. | Retain an `auto` policy scoped by format, JDK, storage, and measured x86 evidence. Do not raise the minimum Java version or generalize the result to unmeasured architectures. |
| 2026-07-19/20 | Does Graal's signed-short pairwise lowering improve a production Q4_0 graph? | [`graal-q4-short-pairwise.md`](graal-q4-short-pairwise.md) records a 22.3% kernel latency reduction and 21.2% median decode gain with exact output hashes. It also records allocation-heavy, globally aggressive, and internal-annotation variants that failed. | Retain the opt-in Graal profile with its inlining setting and shape guard. Keep widened Java execution as the portable default. |
| 2026-07-31 | Can unpacked K-quant weights be reused across prompt rows? | [`q4k-register-tile-20260731.json`](q4k-register-tile-20260731.json), [`q5k-register-tile-20260731.json`](q5k-register-tile-20260731.json), and [`q6k-register-tile-20260731.json`](q6k-register-tile-20260731.json) pair three-fork kernel gates with exact full-model trials. Q4_K and Q5_K gained materially; the Q6_K one-query block traded a smaller speedup for lower fixed-heap collection pressure. | Retain hardware-scoped x86 register tiles. Keep Q6_K's one-query block as the default and the faster two-query block available only to an exact Models profile. |
| 2026-08-30 | Can a two-row Q5_K batch avoid unpacking the same tail twice? | [`q5k-two-query-tail-20260830.json`](q5k-two-query-tail-20260830.json) records exact independent-query and mapped-weight tests plus a three-fork improvement from 0.556 to 0.330 ms/op. | Retain the x86/256-bit two-query tail; leave other architectures on the established path pending evidence. |
| 2026-08-30 | Can standard MXFP4 weights execute efficiently without expanding to F32? | [`mxfp4-w4a8-intel-mac-20260830.json`](mxfp4-w4a8-intel-mac-20260830.json) records the standard E2M1/E8M0 layout, a 7.53x storage reduction, numerical gates, and prepared-activation reuse. | Retain the Java W4A8 primitive. Treat its one-fork Intel result as a kernel screen, not a full-model or cross-platform claim. |
| 2026-08-30 | Is scale lookup, copying weights, or the MXFP4 arithmetic itself the next bottleneck? | [`mxfp4-scale-access-epyc-20260830.json`](mxfp4-scale-access-epyc-20260830.json) records three-fork GPT-OSS expert-shape gates and the official-checkpoint screen. Direct scale access improved both shapes; the heap-copy variant was rejected. | Retain standard-Java scale access and zero-copy mapped weights. Continue optimizing the measured expert hot loop rather than materializing the checkpoint. |
| 2026-09-02 | Can MobileMoE's input-major packed INT4 experts meet the model gate without a native shim? | [`mobilemoe-packed-int4-runtime-layout-20260902.md`](mobilemoe-packed-int4-runtime-layout-20260902.md) rejects BF16 expansion, records 2.76x-13.90x prepared-Q8 kernel gains, and connects them to a 27/27 `PRODUCTION_READY` pure-Java model run. | Keep the mapped INT4 artifact and prepare Q8_0 execution weights by default, with direct packed INT4 as the compact fallback. Carry original-layout signed-INT4 dot-product lowering forward as a concrete JVM request. |

The journal deliberately includes rejected candidates. They prevent a future optimization pass from
repeating a result that already failed correctness, allocation, full-model, or portability gates.

Audit T4.11 (2026-06-06) noted this nuance. The `System.gc()` calls in
the hand-rolled benchmarks have been audited; the only ones that remain
are in `BuildScalabilityBenchmark`, where they're justified for heap-size
measurement (and are commented as such). The latency benchmarks
(`RecallQpsBenchmark`, `QuantizationRecallBenchmark`) no longer call
`System.gc()` between warmup and measurement — the GC pauses they
previously suppressed are part of the latency distribution we want to
publish.
