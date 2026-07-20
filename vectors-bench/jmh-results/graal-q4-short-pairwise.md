# Graal Q4_0 Signed-Short Pairwise Gate

Date: 2026-07-19 (America/Phoenix; measurements completed 2026-07-20 UTC)

This report records the acceptance evidence for the experimental
`vectors.gguf.q4ShortPairwise` kernel. The default remains `false`. The retained deployment
profile requires a post-fix Graal compiler and a larger per-callee exploration limit:

```text
-Djdk.graal.MaximumInliningSize=10000
-Dvectors.gguf.q4ShortPairwise=true
```

It does not require `CompileCommand`, internal JDK annotations, or a higher Java bytecode level.
Vectors continues to compile, test, and publish for Java 25.

To roll back, remove `vectors.gguf.q4ShortPairwise` or set it to `false`; the established widened
kernel remains the default. The Graal inlining limit can then also be removed.

## Environment

- Vectors candidate: `b3aece4`
- Models control: `ab12fff`
- Controlled host: AMD EPYC Milan, 4 physical cores / 8 vCPUs, AVX2, 30 GiB
- JVM: GraalVM CE 25.0.3+9-jvmci-25.1-b19, development build from 2026-07-17
- Vector width: 256 bits
- Model: Qwen3-0.6B-Q4_0, SHA-256
  `da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4`
- Prompt SHA-256: `2db2d875631cc7e3af3f6e4471ae4c9b2b7dfdb31ab561a41ef78182a31532e6`

## Kernel

The candidate preserves Q4_0's signed nibble decode and Q8_0's full signed-byte domain. It widens
both operands to signed shorts, expresses adjacent short multiply-add as the expanded Vector API
graph recognized by Graal's `AMD64SimdPairwiseMultiplyAddNode`, and then combines adjacent integer
pairs into the established four-product lanes. Graal lowers the central graph to `VPMADDWD`.

The production route is admitted only when all of these conditions hold:

- `vectors.gguf.q4ShortPairwise=true` was set before Vectors initializes;
- the active Vector API width is at least 256 bits; and
- the row has at least 32 Q4 blocks, or 1,024 dimensions.

The widened kernel remains active for every other shape. Ten thousand randomized blocks, including
`Byte.MIN_VALUE` Q8 values and nibble-boundary cases, matched the widened kernel lane for lane.

## Kernel Gate

The final production GEMV benchmark used a 1024x2048 Q4_0 matrix, three forks, five one-second
warmups, five three-second measurements, eight persistent workers, and the final two-property
profile above. Neither mode collected a GC.

| Production GEMV | Widened | Short pairwise | Change |
| --- | ---: | ---: | ---: |
| Average time | 0.094 +/- 0.003 ms/op | 0.073 +/- 0.004 ms/op | -22.3% |
| Normalized allocation | 51.67 B/op | 47.22 B/op | no material change |

The 99.9% confidence intervals do not overlap. The result uses the public production entry point,
not only a package-private arithmetic helper.

## Full Model Gate

Qwen used a 2,048-token context, eight threads, batch-32 prompt processing, final-layer K/V-only
prompt rows, two warmups, and three measured 64-token greedy trials in each of four counterbalanced
process pairs. Both modes used `-Djdk.graal.MaximumInliningSize=10000`; the only difference was the
Vectors kernel property.

| Qwen3 Q4_0 metric | Widened | Short pairwise | Change |
| --- | ---: | ---: | ---: |
| Median decode | 28.7519 tok/s | 34.8445 tok/s | +21.19% |
| Median prefill | 70.0064 tok/s | 88.5120 tok/s | +26.43% |
| Median TTFT | 2,220.8 ms | 1,757.5 ms | -20.86% |
| Mean trial CPU | 29,690.8 ms | 23,250.8 ms | -21.69% |

All 12 paired trials matched input-token count, output-token count, and output SHA-256. Peak RSS
was lifecycle-sensitive and varied by several hundred MiB between otherwise equivalent processes;
no memory-capacity claim is made from the process high-water marks.

## Rejected Configurations

- Unsigned-byte/signed-byte 256-bit graphs were exact but shape expansion prevented scalar
  replacement and created roughly 10 KiB per block invocation.
- Signed and offset-corrected 128-bit byte-pairwise graphs were allocation-free only after extra
  activation sums, then remained 37-50% slower than widening.
- The retained signed-short graph without a larger Graal inlining limit allocated approximately
  117 MiB per 1024x2048 GEMV and took about 3.58 ms/op. This is why the Vectors property is not an
  independent tuning switch.
- Raising `TrivialInliningSize`, `MaximumInliningSize`, and the global desired graph budget together
  made full-model performance bimodal and drove otherwise unchanged Qwen processes to about
  9 tok/s. Global aggressive inlining is rejected.
- `CompileCommand=inline` for only the helper methods worked in several processes but remained
  nondeterministic across JMH forks until `MaximumInliningSize` was raised. The commands are not
  part of the final profile.
- `jdk.internal.vm.annotation.ForceInline` on application methods was ignored by this Graal build.
  It left all three candidate forks at about 3.58 ms/op and 117 MiB/op, so no internal JDK API or
  module export is retained.

## Reproduction

Build the Java 25 benchmark JAR:

```bash
./gradlew :vectors-core:check :vectors-bench:jmhJar
```

Run the widened control, then repeat with `vectors.gguf.q4ShortPairwise=true`:

```bash
java -jar vectors-bench/build/libs/vectors-bench-0.1.0-SNAPSHOT-jmh.jar \
  GgufQuantizedMatVecBenchmark.q4_0WithQ8_0Activation \
  -p rows=1024 -p cols=2048 -wi 5 -i 5 -w 1s -r 3s -f 3 -prof gc \
  -jvmArgsAppend \
  '-Djdk.graal.MaximumInliningSize=10000 -Dvectors.gguf.q4ShortPairwise=false'
```
