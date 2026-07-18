# Vector API Pairwise Multiply-Add JVM Baseline

Date: 2026-07-18

This is a controlled comparison of the standalone graphs in
`VectorApiPairwiseMultiplyAddBenchmark`. It verifies the Graal fix before those graphs are
considered for a production quantized model kernel.

## Environment

- Host: KVM, AMD EPYC Milan, 4 physical cores / 8 logical CPUs, AVX2, no AVX-512
- Affinity: benchmark process and forks pinned to logical CPU 2
- JMH: 1.37, average time, 3 forks, 5 x 1-second warmups, 5 x 1-second measurements
- Profiler: `gc`
- HotSpot: Temurin 25.0.3+9-LTS, C2
- Graal: GraalVM CE 25.2.4-dev+9.1, OpenJDK 25.0.3+9, JVMCI 25.1-b19
- Graal development build: `25.2.4-dev-20260717_0119`
- Graal archive SHA-256: `3a4edf22112daa4433556b5aae1ceb707991ef4851ff24952e5a5dd7b5913597`

## Results

| Width | Kernel | HotSpot 25 C2 | Graal development build | Graal speedup | Latency reduction |
| ---: | --- | ---: | ---: | ---: | ---: |
| 128 | signed short pairs to int (`PMADDWD`) | 3.0564 +/- 0.0067 ns/op | 1.2754 +/- 0.0108 ns/op | 2.396x | 58.27% |
| 128 | unsigned byte x signed byte, saturated short (`PMADDUBSW`) | 2.6991 +/- 0.0125 ns/op | 1.4084 +/- 0.0243 ns/op | 1.916x | 47.82% |
| 256 | signed short pairs to int (`VPMADDWD`) | 4.1725 +/- 0.0072 ns/op | 1.3047 +/- 0.0706 ns/op | 3.198x | 68.73% |
| 256 | unsigned byte x signed byte, saturated short (`VPMADDUBSW`) | 4.4337 +/- 0.0177 ns/op | 1.4220 +/- 0.0090 ns/op | 3.118x | 67.93% |

All four measurements reported approximately `10^-5 B/op` and no collections. The score error is
JMH's reported confidence interval.

### Complete 32-byte Q8 dot product

The follow-up benchmark compares the existing four-way byte-to-int widening kernel with a
two-way byte-to-short widening graph that uses the new pairwise matcher. Both calculate one exact
signed 32-byte dot product.

| Kernel | HotSpot 25 C2 | Graal development build | Graal speedup | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Current `B2I` plus `VPMULLD` | 3.361 +/- 0.034 ns/op | 2.598 +/- 0.024 ns/op | 1.294x | 22.70% |
| Pairwise `B2S` plus `VPMADDWD` | 7.280 +/- 0.034 ns/op | 3.157 +/- 0.077 ns/op | 2.306x | 56.63% |

Both full-block kernels reported effectively zero allocation and no collections. Within one
runtime, however, the pairwise formulation is 116.6% slower on HotSpot and 21.5% slower on the
post-fix Graal build. The pairwise micro-operation improvement therefore does not translate into a
better Q8 block kernel on this AMD Zen 3 host.

## Generated Code

The Graal JVMCI compilation contained these byte sequences in the benchmark kernel:

```text
c5 f9 f5 c1       vpmaddwd   xmm0,xmm0,xmm1
c4 e2 71 04 c0    vpmaddubsw xmm0,xmm1,xmm0
c5 fd f5 c1       vpmaddwd   ymm0,ymm0,ymm1
c4 e2 75 04 c0    vpmaddubsw ymm0,ymm1,ymm0
```

The mnemonics were independently decoded with GNU `objdump -D -b binary -m i386:x86-64 -Mintel`.
The corresponding HotSpot C2 compilations contained neither opcode and instead retained the
expanded rearrange, extension, multiply, add, and saturating-add sequence.

For the complete Q8 block, the Graal JVMCI compilation used four `VPMOVSXBW` conversions and two
`VPMADDWD` operations in the pairwise kernel. The faster current kernel used eight `VPMOVSXBD`
conversions and four independent `VPMULLD` operations. This confirms that the slower pairwise score
is not caused by failure to activate the new matcher.

## Interpretation

This proves that Graal commit `0bc546878361` materially closes one compiler gap on AMD64. It also
demonstrates why generated instructions are only an intermediate gate: the exact 32-byte Q8
experiment rejects the pairwise formulation on the tested host even after successful lowering.
No production kernel or runtime-specific dispatch is justified by these results.

The remaining API gap is still material for packed mixed-byte Q4 kernels that need
`VPMADDUBSW` followed by `VPMADDWD`, but that requires a quantization-exact full-block benchmark.
The current Graal source also has no equivalent AArch64 SDOT/UDOT lowering.
