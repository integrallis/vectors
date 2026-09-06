# Soprano Q8 execution-layout and scheduling experiments

Date: 2026-09-04 to 2026-09-05

## Question

Which changes to Vectors' GGUF Q8_0 matrix path improve Soprano speech synthesis, and which
attractive kernel results fail when exercised by the complete model?

The complete qualification and raw model-level measurements live in the Models repository:

- [Soprano Java-native qualification](https://github.com/integrallis/models/blob/main/benchmark-results/audio/soprano-1.1-80m-java/README.md)
- [Q8 scheduling experiment](https://github.com/integrallis/models/blob/main/benchmark-results/audio/soprano-q8-scheduling-20260904/README.md)

This record indexes the Vectors benchmark implementations that produced or reproduce those
decisions. The external Soprano and audio.cpp implementations were oracle and benchmark inputs
only; production parsing, graph execution, sampling, vocoding, streaming, and PCM generation stay
in process.

## Retained experiment sources

| Experiment | Reproducer | Result |
| --- | --- | --- |
| Expand Q8_0 weights once into an owned F32 execution matrix | `GgufQ8DequantizedF32Benchmark`, `SopranoLmQ8F32Benchmark` | Selected for the Soprano vocoder and released as `GgufQ8_0F32Matrix`. |
| Split immutable Q8_0 scales and quant bytes at load time | `SopranoPreparedQ8WeightsBenchmark` | Rejected: 56.249 ms versus 58.374 ms with overlapping confidence intervals; the possible 3.6% gain did not justify 5.9% additional storage or a second production layout. |
| Tile rows and batches while retaining Q8_0 weights and activations | `Q8TiledMatmulBenchmark` | Rejected: the best 4-row by 8-batch tile took 688.0 ms versus 322.0 ms for the production prequantized control, while remaining bit-exact. |
| Lower the row-parallel scheduling threshold for batch-one LM projections | `SopranoLmQ8SchedulingBenchmark` | Rejected by the model gate: isolated projections improved by 2.88x to 3.01x, but repeated executor barriers made complete LM time roughly 40% slower. |

The benchmark-only rejected implementations remain in `vectors-bench` so later optimization work
can rerun the same shapes without reconstructing them from prose. They are not public Vectors APIs
and are not on a production dispatch path.

## Selected result

For the Soprano vocoder's 253 by 2304 by 768 projection, preparing F32 weights once cost 4.560 ms
and expanded one matrix from 1.793 MiB to 6.750 MiB. The parallel F32 kernel took 15.769 ms versus
63.260 ms for the Q8 path, a 4.01x kernel improvement. Across the vocoder's 17 Q8 projections, the
runtime spends 83.72 MiB of additional execution memory.

That trade cleared the complete audio policy. The exact Q8 artifact reached 0.998389315 PCM cosine,
24.923 dB SDR, 676.696 ms p95 time to first audio, and 1.225912 p95 real-time factor on the
controlled Corretto 25 / Intel Xeon 8488C host. The Q8 language-model projections still use the
narrow Models-owned native kernel. The BF16 artifact clears the same policy entirely in Java.

## Why the rejected results matter

The scheduling experiment is the clearest warning against optimizing from isolated matrix timing.
Moving the global threshold made each measured batch-one projection faster, yet the transformer
paid the executor barrier at every projection and became materially slower overall. Likewise,
two-dimensional tiling increased reuse in the inner loop but lost on repeated loads and workspace
overhead at the complete production shape.

The acceptance boundary is therefore the model, not the opcode or microbenchmark. Future Q8 work
must retain exact kernel output and then clear waveform correctness, time to first audio, real-time
factor, and memory together.

## Next JVM-facing gate

The remaining Q8 bottleneck is signed INT8 dot product with INT32 accumulation. Java 25 must widen
byte lanes manually and exposes neither a portable dot-accumulate operation nor a capability query
for VNNI, AArch64 SDOT, or SMMLA lowering. A future JVM or Java-kernel candidate succeeds only if it
beats the current Q8 projection on x86-64 and AArch64 and removes the Soprano shim while preserving
the complete waveform and streaming gates.
