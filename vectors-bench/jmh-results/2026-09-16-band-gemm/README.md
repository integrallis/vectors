# 2026-09-16 — Band GEMM (dequantise-to-F32 panel + FMA tile) vs integer kernels

Branch `exp/band-gemm`. The pre-registration (question and decision rule) was committed in
`370095f` before any A/B run. The sections below it were added afterwards.

## Question

Does a pure-Java "dequantise a cache-sized weight panel to F32, then sweep the activations with a
vector FMA register tile" batched matmul (`GgufBatchedMatmulKernel.BAND_F32`, idea from the
Apache-2.0 qxotic JAM `BandGemm`, re-written here) beat the established integer kernels (Q8
activations, int8/int16 lanes, register tiles) for GGUF Q4_K, Q6_K and Q8_0?

## Decision rule (recorded before running)

The band arm is worth productising only if it beats the existing kernel by **>= 10 %** on at least
one LLM shape on some ISA **without losing > 5 %** elsewhere. Otherwise it stays an experiment.

"LLM shape" = Granite 4.1 3B FFN projections (hidden 2560, FFN 8192): up/gate `8192x2560` and down
`2560x8192`, at batch 1 (decode) and batch 512 (prefill). "Elsewhere" = every other measured
format / shape / batch / thread configuration on the same ISA. A result from one ISA does not decide
for another ISA.

Before any productisation, a separate model-level gate is still required (the band arm is not
numerically identical to the integer kernels; see the correctness section).

---

## Result in one paragraph (measured here, AVX2 only)

**The rule is not met on AVX2: the band arm stays an experiment.** On the one host measured (an
Intel i7-9750H, AVX2, *not* Apple Silicon — see Hardware), the band arm was **5.2x–10.4x faster
than the integer kernels at prefill batch 512** on both Granite FFN shapes for all three formats. That
result appeared in the first multi-threaded run, in a lower-load replicate, in the single-threaded
run, and with mapped weights. It was **0.32x–0.78x as fast at decode (batch 1)** in the replicate, single-threaded and mapped
runs, so it loses by far more than 5 %. The first, heavily loaded run had a single decode cell
above 1.0x (1.11 ± noise). At the smaller existing-benchmark shape (`1024x2048`, batch 2–32),
the two multi-threaded runs contradict each other (for example, Q8_0 batch 32 was 1.30x in run 1
and 0.36x in the replicate), so this host cannot decide those cells. Nothing was measured on NEON
or AVX-512.

## What was measured, read, and believed

- **Measured here:** everything in the tables below, on one x86 AVX2 laptop under heavy,
  uncontrolled load from other workloads (see "Host load"). Correctness bounds were also measured
  here (tests).
- **Read, not measured:** qxotic's comments on Zen 5 (e.g. Q8_0 band vs tile 1400 → 2700 GF/s at
  16T; nibble unpack 4.9 vs 16 Gelem/s in int lanes). None of those numbers was reproduced here.
- **Believed, not measured:** the 4x4-on-32-register default (AVX-512 C2, AArch64). That comes
  from counting registers. The local 3x3 vs 4x4 runs on AVX2 contradict each other (see Tile
  shape).

## Hardware and runtime

| | |
|---|---|
| Host | MacBook Pro, **Intel Core i7-9750H** (Coffee Lake, 6C/12T), `x86_64`, **AVX2 + FMA, no AVX-512** |
| Caches | L1d 32 KiB, L2 256 KiB per core; 32 GiB RAM |
| OS | macOS (Darwin 25.6.0 x86_64) |
| JDK | Temurin 25.0.3+9, HotSpot C2 |
| Species | `vectors.maxBits` default 256 → FloatVector 256-bit, 8 lanes; `HAS_FAST_VECTOR_FMA=true` |
| GGUF executor | `persistent`, 12 threads, 2 chunks/thread (same for both arms) |
| JVM flags | `--add-modules jdk.incubator.vector -Xmx8g -Xms8g -XX:+AlwaysPreTouch -XX:+UseG1GC` (+ `-Dvectors.gguf.parallel=false` for ST, `-Dvectors.gguf.band.tile=...` for tile runs) |
| JMH | 1.37, 1 fork; 3 warmup x 2 s and 5 measurement x 2 s (ST LLM phase: 2 x 1 s and 3 x 1 s) |
| Band config | `band-f32(tile=3x3,kc=512,panelKb=256,lanes=8)` (self-reported by each fork in the tile rerun) |

The brief assumed this Mac was Apple Silicon (NEON, 128-bit). It is not: `uname -m` = `x86_64`,
`sysctl.proc_translated` is unset (not Rosetta), and the CPU brand is Intel. **No NEON result exists.**
128-bit lanes were exercised only for correctness (`-Dvectors.maxBits=128` probe), not for
performance.

Full host facts per run are in `raw/<label>/host.txt`; each fork's `VectorRuntimeCapabilities` line
is in `raw/<label>/*.txt`.

### Host load (limits every number here)

The laptop was shared with other sessions' JVMs (a memory server, Gradle daemons, IDEs, a VM). The
1-minute load average was **246–658** during the bandwidth, dequant and first multi-threaded phases,
and 20–45 during the later phases and the replicate (`raw/*/load.txt`). JMH error bars often
exceed the score. So:

- read every number as a **screen**, not as a publishable measurement;
- trust only effects that are large compared with their error and that repeat across runs (prefill
  batch 512, decode batch 1);
- treat mid-batch cells at `1024x2048` and the tile comparison as **unresolved**.

## Correctness (measured here; tests in `vectors-core`)

`GgufBandGemmTest` compares the band arm with an **exact double-precision reference**. The
reference dequantizes each element from the block bytes with its own code, independent of both
kernels.

- **Accepted band error:** `|band − exact| ≤ 1e-4 · Σ|w·x| + 1e-6`. Measured maximum:
  **5.0e-8 · Σ|w·x|**.
  - The bound is relative to the size of the terms, not to the result, which can cancel towards 0.
  - First-order float32 rounding in lane-wise accumulation is about `(k/lanes)·eps`, which is
    2.4e-4 at k=8192 with 4 lanes. In practice it is about `sqrt(k)·eps`.
  - 1e-4 is tight enough that any indexing or bit-extraction bug fails, because such a bug moves a
    result by whole terms.
  - Mutation checks run here: breaking the Q4_K high-nibble shift, the Q6_K sub-block scale index,
    or one 3x3 tile cross-term fails 23, 23 and 17 of 32 tests respectively.
- **Integer kernels (Panama and scalar reference provider) vs exact:**
  - They cannot match the band arm bit-for-bit, because they quantize each activation block to int8
    first.
  - They are held to the analytic activation-rounding bound `Σ|w|·scale_block/2`, with
    `scale = max|x|/127` per Q8_K (256) or Q8_0 (32) block.
  - Measured worst case: **0.187 of that bound**, and at most 2.7e-3 · Σ|w·x|.
  - This validates the test's dequantization against the shipped kernels' reading of the bit layout.
    It also bounds `|band − integer|`, which is asserted as well.
- **Shapes:**
  - Decode batch 1 and rows=1.
  - Batches 2, 3, 4, 5, 7 and 11 (not multiples of 3 or 4).
  - Rows not a multiple of the band.
  - Columns below, equal to, and not a multiple of the 512-element column block.
  - Several panels, and the parallel path (Q4_K/Q6_K `5x512x2560`, Q8_0 `3x1600x2560`).
- **Switch observability:**
  - A fresh-JVM probe runs with `-Dvectors.gguf.batchedMatmulKernel=band` at 256-bit/4x4,
    128-bit/3x3 and 128-bit/4x4.
  - It checks that the capabilities report `band-f32(tile=…,lanes=…)`, and that the default entry
    point output for all three formats is bit-identical to the explicit band arm and differs from
    the integer arm.
  - It also checks band vs exact at a `2x3x8192` shape.
  - In-process, the default is `integer` and bit-identical to the explicit integer arm.
- **Numerical consequence:** switching the arm changes model logits by the size of the integer
  path's activation-rounding error, in the direction of the exact float result. This needs a
  model-level gate before any default change.
- **Determinism:** the band reduction sums lanes in a fixed order instead of `reduceLanes(ADD)`, to
  avoid the interpreter-vs-C2 tree difference recorded on 2026-09-14. A cold/hot bit-identity probe
  was **not** written for the band arm (follow-up).

## Structural test (measured: red → green)

`VectorSpeciesConstantStructureTest` scans the bytecode of every vectors-core main class that
references `jdk.incubator.vector`. It fails on a `VectorSpecies` method parameter, and on a species
held in an instance or non-final static field.

- **Red on main 5ce9b4c** (empty allow-list): exactly one violation,
  `PanamaConstants#preferredSpecies(VectorSpecies)VectorSpecies`. This is the capping helper that
  builds the static final species at class init.
- **Green:** that entry is allow-listed with a comment. Stale allow-list entries also fail.
- The detector is tested against positive and negative fixtures, and the scan must include
  `PanamaVectorUtilSupport` and `GgufBandGemm`, so a green run cannot be vacuous.
- Follow-up: none required for hot kernels. The only violation runs once at class init.

## Results

All numbers are ms/op (lower is better). The `band speedup` column is integer ms / band ms.

- **Weights GB/s:** the quantized matrix bytes divided by the op time. Both arms read the whole
  matrix once per op.
- **% bandwidth:** that rate as a share of `MemoryBandwidthBenchmark` (1-thread probe for ST rows,
  all-threads probe for MT rows).
- **Mul-adds/s:** `batch·rows·cols / s`.

The full tables, including the existing `1024x2048` shape, are in `summary-*.md`.

### Memory bandwidth roofline (measured under load 246–635; expect an underestimate)

| probe | threads | GB/s |
|---|---:|---:|
| `sequentialLongSum` | 1 | 7.44 |
| `sequentialByteXor` | 1 | 6.95 |
| `sequentialLongSum` | 12 | 15.23 |
| `sequentialByteXor` | 12 | 15.03 |

### Granite 4.1 3B FFN shapes — multi-threaded (12 threads), replicate at load 27–45 (`raw/intel-i7-9750h-avx2-rep2`)

| format | shape | batch | integer | band | band speedup | mul-adds/s int / band (G) |
|---|---|---:|---:|---:|---:|---:|
| Q4_K | 8192x2560 | 1 | 8.39 ± 20.5 | 11.86 ± 2.1 | **0.71x** | 2.50 / 1.77 |
| Q4_K | 8192x2560 | 512 | 1432.6 ± 188.5 | 227.8 ± 28.8 | **6.29x** | 7.50 / 47.14 |
| Q4_K | 2560x8192 | 1 | 6.63 ± 1.2 | 10.75 ± 1.9 | **0.62x** | 3.16 / 1.95 |
| Q4_K | 2560x8192 | 512 | 1386.8 ± 114.4 | 249.8 ± 20.2 | **5.55x** | 7.74 / 42.98 |
| Q6_K | 8192x2560 | 1 | 13.64 ± 17.9 | 17.54 ± 3.3 | **0.78x** | 1.54 / 1.20 |
| Q6_K | 8192x2560 | 512 | 2079.4 ± 201.0 | 233.1 ± 25.3 | **8.92x** | 5.16 / 46.06 |
| Q6_K | 2560x8192 | 1 | 7.06 ± 3.4 | 16.32 ± 2.0 | **0.43x** | 2.97 / 1.29 |
| Q6_K | 2560x8192 | 512 | 2015.0 ± 208.1 | 258.2 ± 5.4 | **7.80x** | 5.33 / 41.58 |
| Q8_0 | 8192x2560 | 1 | 4.05 ± 0.6 | 7.34 ± 1.4 | **0.55x** | 5.17 / 2.86 |
| Q8_0 | 8192x2560 | 512 | 1501.1 ± 117.7 | 211.3 ± 23.9 | **7.10x** | 7.15 / 50.82 |
| Q8_0 | 2560x8192 | 1 | 4.17 ± 0.4 | 6.21 ± 0.4 | **0.67x** | 5.03 / 3.38 |
| Q8_0 | 2560x8192 | 512 | 1556.8 ± 203.1 | 241.1 ± 8.7 | **6.46x** | 6.90 / 44.54 |

The first multi-threaded run (load 120–658) gave the same direction: batch 512 5.18x–10.43x, and
batch 1 0.47x–1.11x (the one decode cell above 1.0x was Q6_K `8192x2560`, 22 ± 43 ms, which is noise).

### Granite FFN shapes — single-threaded (`-Dvectors.gguf.parallel=false`), load ≈20

| format | shape | batch | integer | band | band speedup | % 1T bandwidth int / band |
|---|---|---:|---:|---:|---:|---:|
| Q4_K | 8192x2560 | 1 | 29.2 | 91.3 | 0.32x | 5.4 / 1.7 |
| Q4_K | 8192x2560 | 512 | 9073 | 1325 | 6.85x | 0.0 / 0.1 |
| Q4_K | 2560x8192 | 1 | 27.9 | 77.6 | 0.36x | 5.7 / 2.0 |
| Q4_K | 2560x8192 | 512 | 8973 | 1269 | 7.07x | 0.0 / 0.1 |
| Q6_K | 8192x2560 | 1 | 46.7 | 106.0 | 0.44x | 5.0 / 2.2 |
| Q6_K | 8192x2560 | 512 | 14491 | 1408 | 10.29x | 0.0 / 0.2 |
| Q6_K | 2560x8192 | 1 | 41.9 | 90.1 | 0.47x | 5.5 / 2.6 |
| Q6_K | 2560x8192 | 512 | 15972 | 1374 | 11.62x | 0.0 / 0.2 |
| Q8_0 | 8192x2560 | 1 | 19.6 | 33.3 | 0.59x | 15.3 / 9.0 |
| Q8_0 | 8192x2560 | 512 | 8267 | 2002 | 4.13x | 0.0 / 0.1 |
| Q8_0 | 2560x8192 | 1 | 21.2 | 32.9 | 0.65x | 14.1 / 9.1 |
| Q8_0 | 2560x8192 | 512 | 14827 | 1740 | 8.52x | 0.0 / 0.2 |

This phase used 1 op per iteration and 3 iterations, so its error bars are ±40–140 %; see
`summary-intel-i7-9750h-avx2.md`. Its direction matches the multi-threaded runs.

### Mapped weights (Q4_K, multi-threaded; exercises the x86 mapped long-offset policy)

| shape | batch | integer | band | band speedup |
|---|---:|---:|---:|---:|
| 8192x2560 | 1 | 6.05 | 17.11 | 0.35x |
| 8192x2560 | 512 | 1565.3 | 294.8 | 5.31x |
| 2560x8192 | 1 | 8.81 | 20.97 | 0.42x |
| 2560x8192 | 512 | 1959.3 | 308.9 | 6.34x |

### Component ablation — band dequantization alone (single thread, load 635–658)

| format | shape | F32 elements/s | quantized GB/s | % 1T bandwidth |
|---|---|---:|---:|---:|
| Q4_K | 8192x2560 | 87 M | 0.049 | 0.7 |
| Q4_K | 2560x8192 | 125 M | 0.070 | 0.9 |
| Q6_K | 8192x2560 | 95 M | 0.078 | 1.0 |
| Q6_K | 2560x8192 | 119 M | 0.098 | 1.3 |
| Q8_0 | 8192x2560 | 365 M | 0.387 | 5.2 |
| Q8_0 | 2560x8192 | 486 M | 0.516 | 6.9 |

Dequantization is compute-bound, not bandwidth-bound: it streams 1–7 % of one core's measured
bandwidth.

- **K-quants:** the scalar dequant loops (a bulk copy, then per-group loops left to C2 SuperWord)
  run at about 11 ns per element. That points to little or no auto-vectorization, but no
  disassembly was taken to confirm it.
- **Decode:** in the single-threaded Q4_K `8192x2560` decode cell, dequantization of the whole
  matrix (≈240 ms in this ablation under heavier load) exceeds the whole band op (91 ms at load 20).
  Given the load difference, the only safe reading is that **dequant is the dominant decode
  cost**. This is the first thing to fix before the band arm could compete at decode.
- **Sweep:** the sweep itself reached 42–51 G mul-adds/s on 12 threads at batch 512 (replicate),
  against 5–8 G for the integer kernels at the same shapes.

### Tile shape on AVX2 (band arm only) — unresolved

| format | shape | batch | 3x3 ms/op | 4x4 ms/op | 4x4 vs 3x3 (rerun, self-reported) | 4x4 vs 3x3 (first run, flag only) |
|---|---|---:|---:|---:|---:|---:|
| Q4_K | 8192x2560 | 1 | 9.32 | 9.46 | 0.99x | 0.83x |
| Q4_K | 8192x2560 | 512 | 306.3 | 515.6 | 0.59x | 1.24x |
| Q4_K | 2560x8192 | 1 | 9.79 | 12.21 | 0.80x | 0.84x |
| Q4_K | 2560x8192 | 512 | 275.4 | 439.5 | 0.63x | 1.04x |
| Q8_0 | 8192x2560 | 1 | 6.27 | 10.15 | 0.62x | 1.02x |
| Q8_0 | 8192x2560 | 512 | 236.8 | 303.5 | 0.78x | 0.89x |
| Q8_0 | 2560x8192 | 1 | 5.72 | 19.62 | 0.29x | 1.08x |
| Q8_0 | 2560x8192 | 512 | 199.1 | 239.0 | 0.83x | 1.07x |

The two runs disagree in sign, so **there is no conclusion here**.

- The rerun printed `band-f32(tile=4x4,…)` from inside each fork.
- The first run only carried the `-D` flag, before the arm reported its configuration when selected
  explicitly. Its files are kept as `raw/intel-i7-9750h-avx2/tile-*.unreported.*`.
- The 3x3 default on 256-bit species stays in place. It rests on the register count (13 live
  vectors fit in 16 YMM), not on this data.

## Decision (against the pre-registered rule)

| ISA | LLM-shape win ≥ 10 %? | Loss > 5 % elsewhere? | Verdict |
|---|---|---|---|
| x86 AVX2 (i7-9750H, 256-bit) — measured | yes: batch 512, all formats, both shapes, 5.2x–10.4x MT | yes: batch 1 in every run (0.32x–0.78x outside the loaded first run) | **fails → stays an experiment** |
| x86 AVX-512 | not measured | not measured | pending |
| AArch64 NEON | not measured | not measured | pending |

The rule was written for a single kernel replacing another, and it rejects that replacement here.
The data suggests a different candidate that the rule did not cover:
**dispatch by batch size**, with integer for decode and small batches and band for prefill.

- Evaluating it needs its own pre-registered rule and a measured crossover batch per format and ISA.
- On this host the crossover fell somewhere between 1 and 8 at `1024x2048` single-threaded. At
  batch 8 the band arm was 1.71x–2.22x faster; batches 2 and 4 were not run ST.
- The multi-threaded mid-batch cells are unresolved.
- It also needs the model-level numerical gate, because band and integer outputs differ.

## Commands

Local run (this report):

```bash
./gradlew :vectors-bench:jmhJar
vectors-bench/jmh-results/2026-09-16-band-gemm/run.sh intel-i7-9750h-avx2          # all phases
vectors-bench/jmh-results/2026-09-16-band-gemm/run.sh intel-i7-9750h-avx2-rep2 ab-mt
vectors-bench/jmh-results/2026-09-16-band-gemm/run.sh intel-i7-9750h-avx2-tile tile
python3 vectors-bench/jmh-results/2026-09-16-band-gemm/summarize.py \
  vectors-bench/jmh-results/2026-09-16-band-gemm/raw/intel-i7-9750h-avx2
```

`run.sh` records host facts, the JVM flags, and the load average before and after each phase. Each
phase is plain JMH, so any single cell can be rerun directly, for example:

```bash
java -jar vectors-bench/build/libs/vectors-bench-*-jmh.jar GgufBandGemmAbBenchmark \
  -p format=Q4_K -p shape=8192x2560 -p batchSize=1,512 -f 1 -wi 3 -i 5 -w 2 -r 2 \
  -jvmArgs "--add-modules jdk.incubator.vector -Xmx8g -Xms8g -XX:+AlwaysPreTouch -XX:+UseG1GC" \
  -rf json -rff out.json
```

### Ready-to-run on an x86 AVX-512 host (also runs the AVX2 species on the same box)

Use a quiet, dedicated host (no other benchmark or build running), with SMT and frequency behaviour
recorded.

```bash
git clone git@github.com:integrallis/vectors.git && cd vectors && git checkout exp/band-gemm
./gradlew :vectors-bench:jmhJar
R=vectors-bench/jmh-results/2026-09-16-band-gemm
# 512-bit species (4x4 band tile by default), all phases:
MAXBITS=512 $R/run.sh "$(hostname)-avx512" bandwidth dequant ab-mt ab-st tile mapped
# library-default 256-bit species on the same host (3x3 band tile by default):
$R/run.sh "$(hostname)-avx2" bandwidth dequant ab-mt ab-st tile mapped
python3 $R/summarize.py $R/raw/"$(hostname)-avx512"
python3 $R/summarize.py $R/raw/"$(hostname)-avx2"
```

- **Threads:** optionally pin, e.g. `numactl --cpunodebind=0 --membind=0 $R/run.sh …`, and add
  `THREADS=<physical cores>` so the GGUF executor does not oversubscribe SMT siblings.
- **Before trusting a cell:** raise `-f` to 3, following the repository convention for decisive
  gates, by editing `run.sh`'s `-f 1`.
- **Other runs still missing:** an Apple Silicon run (`$R/run.sh "$(hostname)-neon"`) is needed for
  the NEON row. It picks the 4x4 tile on `aarch64` by default.

## Follow-ups

1. **Vector API dequantization** for Q4_K and Q6_K (int-lane nibble unpack, width-generic
   `castShape(species, part)` as in qxotic; measure it, don't import their numbers). Dequant is the
   dominant decode cost here.
2. **Batch-size dispatch** (integer below a measured crossover, band above it). This needs its own
   pre-registered rule, a crossover sweep (batch 1, 2, 4, 8, 16, 32, 64) per format and ISA, and a
   model-level prefill gate.
3. **Cold/hot determinism probe** for the band arm, like `Q4KColdHotDeterminismProbe`.
4. **Tile shape and `kc`/`panelKb`** on a quiet host. The AVX2 tile result is unresolved, and
   AVX-512/NEON were not measured.
5. **Variants not tried:** qxotic's interleaved packing, native scratch with the `GLOBAL` absolute
   segment trick, and quarter-panel tail scheduling. Each is a separate arm and needs a separate
   switch.

## Files

- `summary-genoa-avx512-512bit.md`, `summary-genoa-256-on-avx512.md`: Genoa bandwidth, dequant, ST
  and crossover tables. `crossover.py`: T derivation.
- `model-check/`: pre-registration 2 model-level check (`run-model-check.sh`, `ModelCheck.java`,
  `compare.py`).

- `run.sh`: phase runner. `summarize.py`: tables and the decision rule.
- `summary-intel-i7-9750h-avx2.md`: bandwidth, dequant, MT and ST at both shapes, mapped.
- `summary-intel-i7-9750h-avx2-rep2.md`: MT replicate at lower load.
- `summary-intel-i7-9750h-avx2-tile.md`: tile rerun with self-reported configuration.
- `raw/<label>/*.json|*.txt`: JMH output, per-fork capability lines, `host.txt`, `load.txt`.

## Pre-registration 2: batch-dispatched hybrid (written 2026-09-17T01:05Z, before any run below)

The first rule (one kernel for every shape) failed on AVX2: band won prefill by 5.5x–8.9x and lost
decode by 22–57 %. The candidate the data suggests is a **dispatch**: integer kernels below a batch
threshold, band at or above it. Decided by this rule, fixed before the runs:

- **Hosts:** Hetzner cpx62-class AMD EPYC Genoa (16 shared vCPU, AVX-512 + VNNI), idle; one run with
  512-bit species (`MAXBITS=512`) and one with the library default 256-bit species on the same
  silicon (labelled `256-on-avx512`, not AVX2 hardware). Two forks per cell (`crossover` phase).
- **Crossover:** for each format (Q4_K, Q6_K, Q8_0) and Granite FFN shape (8192x2560, 2560x8192), the
  threshold T is the smallest batch in {1,2,4,8,16,32,64,128,512} at which band's mean beats integer's
  mean by more than both error bars, and stays ahead at every larger batch.
- **Accept the dispatch** if, on both hosts' runs: (a) at batch 512 band beats integer by ≥ 10 % for
  all six format × shape cells; (b) a single T per format works for both shapes (dispatch below T is
  integer, so no configuration can lose more than measurement noise); (c) T ≤ 64, so real prefill
  chunks take the band path.
- **Model-level check before any default change:** pure-Java Granite 4.1 3B prefill tok/s at context
  2048 improves ≥ 10 % with the dispatch on, and greedy continuations (64 tokens) of the frozen
  answerability window's first 20 SQuAD cases are identical to the integer path in ≥ 19 of 20
  (band is the exact float result; integer differs by activation-rounding error, so small
  divergences are expected and must be reported, not hidden).
- **Otherwise** the band kernel stays an explicit opt-in experiment.

---

## Pre-registration 2 — kernel-level results (EPYC Genoa, measured on two hosts, run by the coordinator)

Everything in this section was **measured** on two Hetzner cpx62-class VMs. The runs used this
branch at `228003ac`, with the pre-registered `crossover` phase plus the bandwidth, dequant and
single-thread phases. Raw directories, copied in whole:

- `raw/genoa-avx512-512bit/` (`MAXBITS=512`)
- `raw/genoa-256-on-avx512/` (library default 256-bit species on the same AVX-512 silicon; this is
  **not** AVX2 hardware)

Generated tables are in `summary-genoa-*.md`, produced by `summarize.py` and `crossover.py`.
`crossover.py` is the coordinator's `cross.py` parameterised by path. It reproduces exactly the T
values in the coordinator's message; re-run here from the raw JSON.

### Hosts

| | genoa-avx512-512bit | genoa-256-on-avx512 |
|---|---|---|
| CPU (as reported by the guest) | AMD EPYC-Genoa, 16 vCPU (1 thread/core), KVM | same class, separate VM |
| ISA flags | avx2, fma, avx512f/dq/cd/bw/vl, avx512_vnni, avx512_bf16, vbmi2, bitalg, vpopcntdq | same |
| Caches (guest-reported, virtualised) | L1d 512 KiB total, L2 16 MiB total, L3 32 MiB | same |
| JDK | Temurin 25.0.4.1+1 | same |
| JVM | `-Xmx8g -Xms8g -XX:+AlwaysPreTouch -XX:+UseG1GC -Dvectors.maxBits=512` | same without maxBits |
| Band config | tile 4x4 (512-bit default), kc 512, panel 256 KiB, 16 lanes | tile 3x3, 8 lanes |

- **Load:** 1-minute load average about 1–3 at the start. It was about 14 during the crossover
  sweep, which is the benchmark's own 16 GGUF executor threads. See `raw/*/load.txt`.
- **Crossover:** multi-threaded (16 executor threads), 2 forks, 3 × 2 s warmup, 5 × 2 s measurement.
- **Single-thread phases:** 1 fork, 2 × 1 s warmup, 3 × 1 s measurement.

### Crossover (speedup = integer ms / band ms; `*` = band ahead beyond both error bars)

512-bit species:

| format | shape | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 512 | T |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4_K | 2560x8192 | 0.38 | 0.87 | 1.18* | 2.20* | 3.57* | 4.86* | 6.41* | 8.63* | 9.92* | 4 |
| Q4_K | 8192x2560 | 0.38 | 0.80 | 1.12* | 2.15* | 3.31* | 4.38* | 5.71* | 6.46* | 6.88* | 4 |
| Q6_K | 2560x8192 | 0.30 | 0.57 | 0.60 | 1.22* | 2.06* | 3.48* | 5.10* | 7.60* | 9.20* | 8 |
| Q6_K | 8192x2560 | 0.30 | 0.52 | 0.63 | 1.17 | 1.86* | 3.12* | 4.92* | 6.10* | 10.18* | 16 |
| Q8_0 | 2560x8192 | 0.62 | 1.29* | 1.67* | 2.45* | 3.39* | 4.09* | 4.84* | 5.91* | 6.37* | 2 |
| Q8_0 | 8192x2560 | 0.45 | 0.75 | 1.47* | 2.30* | 3.78* | 5.15* | 4.25* | 4.73* | 7.02* | 4 |

256-bit species on the same silicon:

| format | shape | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 512 | T |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4_K | 2560x8192 | 0.43 | 0.86 | 1.17* | 1.16* | 2.06* | 3.97* | 5.23* | 6.25* | 7.44* | 4 |
| Q4_K | 8192x2560 | 0.43 | 0.85 | 1.20* | 2.02* | 3.18* | 5.10* | 6.05* | 6.46* | 7.21* | 4 |
| Q6_K | 2560x8192 | 0.30 | 0.54 | 0.61 | 0.60 | 1.11 | 2.18* | 3.31* | 5.01* | 7.24* | 32 |
| Q6_K | 8192x2560 | 0.28 | 0.57 | 0.60 | 1.16* | 2.00* | 3.44* | 4.50* | 5.75* | 7.24* | 8 |
| Q8_0 | 2560x8192 | 0.58 | 1.43* | 1.85* | 1.54* | 2.23* | 3.00* | 3.87* | 5.06* | 6.60* | 2 |
| Q8_0 | 8192x2560 | 0.73 | 1.16* | 1.67* | 2.06* | 3.15* | 3.62* | 3.76* | 4.15* | 6.10* | 2 |

**Verdict against pre-registration 2 (kernel half):**

- **(a) Passes.** Batch 512 is at least 1.10x in all 12 cells (6.10x–10.18x).
- **(b) Passes.** Taking the strictest cell across both runs and both shapes gives one threshold
  per format:
  - **Q4_K T = 4**;
  - **Q6_K T = 32** (256-bit, `2560x8192`; the other cells were 8 and 16);
  - **Q8_0 T = 4** (512-bit, `8192x2560`).
- **(c) Passes.** Every T is at most 64.

**Caveats:**

- **Cells between the per-shape crossovers:** below its own crossover a cell may not be clearly
  ahead, but the strictest-cell T keeps every such cell on the integer arm. Every configuration at
  or above T is band-ahead beyond error bars in all four runs (2 hosts × 2 shapes). So none of the
  measured Granite FFN-shape configurations loses to its arm choice by more than noise.
- **Other shapes:** attention projections (2560x2560, 2560x512 KV) and other models were **not**
  measured. The model-level check covers them only in aggregate.

### Genoa: sequential read bandwidth (`MemoryBandwidthBenchmark`)

| probe | 512-bit host | 256-bit host |
|---|---:|---:|
| `sequentialLongSum`, 1 thread | 33.4 GB/s | 31.6 GB/s |
| `sequentialByteXor`, 1 thread | 33.8 GB/s | 31.0 GB/s |
| `sequentialLongSum`, 16 threads (total) | 344 GB/s | 267 GB/s |
| `sequentialByteXor`, 16 threads (total) | 373 GB/s | 309 GB/s |

The 16-thread totals are **as measured, not believed**. They are larger than a 16-vCPU slice of a
shared host plausibly owns, and the probe's multi-thread mode (every thread scanning the same 2 GiB
segment) may be partly measuring the cache and prefetcher rather than DRAM. Use the 1-thread rows as
the roofline.

### Genoa: band dequantization alone (single thread)

| format | shape | 512-bit: M F32 elem/s (quantized GB/s, % 1T bandwidth) | 256-bit: same |
|---|---|---:|---:|
| Q4_K | 8192x2560 | 1323 (0.744, 2.2 %) | 1357 (0.763, 2.4 %) |
| Q4_K | 2560x8192 | 1359 (0.764, 2.3 %) | 1336 (0.751, 2.4 %) |
| Q6_K | 8192x2560 | 607 (0.498, 1.5 %) | 613 (0.503, 1.6 %) |
| Q6_K | 2560x8192 | 619 (0.508, 1.5 %) | 613 (0.503, 1.6 %) |
| Q8_0 | 8192x2560 | 3779 (4.015, 12.0 %) | 3705 (3.936, 12.5 %) |
| Q8_0 | 2560x8192 | 3764 (3.999, 12.0 %) | 3687 (3.918, 12.4 %) |

Dequantization does not depend on the species width, because the dequant loops are scalar and
width-independent. Q6_K dequant is about half the speed of Q4_K dequant, which fits Q6_K's higher
threshold.

The same Q4_K dequant measured 87–125 M elem/s on the loaded AVX2 laptop above. That number was
dominated by host load and is superseded by these.

### Genoa: single-threaded A/B (`-Dvectors.gguf.parallel=false`), Granite FFN shapes

| format | shape | batch | 512-bit int / band ms | speedup | 256-bit int / band ms | speedup |
|---|---|---:|---:|---:|---:|---:|
| Q4_K | 8192x2560 | 1 | 6.56 / 18.56 | 0.35x | 6.72 / 18.58 | 0.36x |
| Q4_K | 8192x2560 | 512 | 2743 / 296 | 9.27x | 2784 / 290 | 9.59x |
| Q4_K | 2560x8192 | 1 | 6.76 / 18.52 | 0.37x | 6.79 / 17.33 | 0.39x |
| Q4_K | 2560x8192 | 512 | 2738 / 303 | 9.03x | 2842 / 334 | 8.51x |
| Q6_K | 8192x2560 | 1 | 10.09 / 37.62 | 0.27x | 10.45 / 38.09 | 0.27x |
| Q6_K | 8192x2560 | 512 | 2869 / 310 | 9.25x | 2869 / 332 | 8.63x |
| Q6_K | 2560x8192 | 1 | 10.41 / 38.32 | 0.27x | 10.13 / 37.53 | 0.27x |
| Q6_K | 2560x8192 | 512 | 2827 / 323 | 8.76x | 2861 / 353 | 8.10x |
| Q8_0 | 8192x2560 | 1 | 4.68 / 9.43 | 0.50x | 5.12 / 10.11 | 0.51x |
| Q8_0 | 8192x2560 | 512 | 2001 / 284 | 7.05x | 2370 / 310 | 7.66x |
| Q8_0 | 2560x8192 | 1 | 4.73 / 9.82 | 0.48x | 5.09 / 8.85 | 0.57x |
| Q8_0 | 2560x8192 | 512 | 1940 / 287 | 6.75x | 2098 / 314 | 6.69x |

- **Existing shape (`1024x2048`, single thread):** band was 0.28x–0.67x at batch 1, 1.13x–2.19x at
  batch 8, and 3.30x–6.15x at batch 32. See `summary-genoa-*.md`.
- **Single-thread gain:** band sweeps reached 30–38 G mul-adds/s on one core at batch 512, against
  3.7–5.5 G for the integer kernels.

## Batch dispatch implementation (`GgufBatchedMatmulKernel.BATCH_DISPATCH`)

- **Selection:** `-Dvectors.gguf.batchedMatmulKernel=dispatch`, or pass `BATCH_DISPATCH` to the
  explicit overloads.
- **Default:** stays `integer` until the model-level check passes.
- **Thresholds:** fixed to the pre-registered values and deliberately **not** configurable:
  Q4_K 4, Q6_K 32, Q8_0 4. "Batch" means the activation rows of the call.
- **Capabilities string:**
  `dispatch(band-at-batch>=Q4_K:4,Q6_K:32,Q8_0:4;band=band-f32(tile=…,kc=512,panelKb=256,lanes=…))`.

### Coverage: which calls the switch reaches

Models' pure-Java backend (origin/main `167a8abd`, `TensorOps`) calls more than the three
single-matrix entry points. The call sites read there are:

- `ggufQ4_KQ8_KDualBatchedMatmul`;
- the mixed `ggufQ4_KQ4_KQ6_KQ8_KTripleBatchedMatmul`;
- `ggufQ8_0Q8_0Dual/TripleBatchedMatmul`;
- the Q6_K overload that takes a `GgufQ6BatchedKernel`.

A switch on the single-matrix calls alone would have left most Granite projections on the integer
arm, and the model-level check would have measured a toggle that barely ran. So the mode now
applies to all of these calls:

- **Grouped calls:** each matrix is routed by its own format.
- **Mixed Q4_K/Q4_K/Q6_K triple:** for batch 4 to 31, the Q4_K pair runs band and the Q6_K matrix
  runs the single-matrix integer kernel. That costs one extra Q8_K quantization pass, and the Q6_K
  output is bit-identical to the grouped integer path (tested).
- **Q6_K tile overload:** it now follows the process mode, and the tile choice applies only when the
  integer arm runs. Before this change, that overload always ran integer in `band` mode.
- **Not covered:** the prequantized `…Rows` entry points (`BatchedMatmulRows`,
  `BlockMajorBatchedMatmulRows`). They receive already-quantized activations and cannot take the
  band path.

### Observability

- Every routed matrix is counted per format and arm, as calls and activation rows.
- `GgufBatchedMatmulKernel.routingReport()` returns the counts, and
  `-Dvectors.gguf.batchedMatmulKernel.report=true` prints them at JVM exit:
  `vectors-gguf-batched-matmul-routing mode=dispatch Q4_K integer=c/r band=c/r Q6_K … Q8_0 …`.
- The model-check script refuses a run in which the dispatch arm reports zero band rows, or the
  integer arm reports any.

### Tests (`GgufBatchDispatchTest`, `GgufBatchDispatchProbe`)

All were written before the implementation; they failed to compile (66 errors) before it existed.

- **Thresholds:** equal to the pre-registered values, and the capabilities string carries them.
- **Predicate:** integer at T−1, band at T.
- **Single-matrix calls:** at T−1 and T for each format, output is bit-identical to the explicitly
  selected arm, and the routing counts are exact.
  - The result is within that arm's documented tolerance of the exact double reference.
  - Band tolerance: 1e-4 · Σ|w·x|. Integer tolerance: the analytic Q8 activation-rounding bound.
- **Grouped calls:**
  - The Q4_K dual and mixed triple are checked at batches 3, 4, 31 and 32, each matrix matching its
    arm.
  - The Q8_0 dual and triple are checked at batches 3 and 4.
- **Fresh-JVM probe:** runs with the property set to `dispatch` through the property-default entry
  points, including the Q6_K tile overload that Models calls. It checks the exit report's exact
  counts.

## Pre-registration 2 — model-level check (prepared here, not run here)

This laptop is the loaded AVX2 i7 and was not used for the check, as instructed. The check is
scripted to run on a Genoa host with one command:

```bash
git clone git@github.com:integrallis/vectors.git && cd vectors && git checkout exp/band-gemm
bash vectors-bench/jmh-results/2026-09-16-band-gemm/model-check/run-model-check.sh
# optional: EXTRA_JVM="-Dvectors.maxBits=512"   PREFILL_REPS=7   ADAPTER_DIR=<answerability aLoRA dir>
```

What it does (`model-check/run-model-check.sh`, `ModelCheck.java`, `compare.py`):

1. **Build.** Clones Models at `167a8abdd662af8d89a821a1bf2d23980e1092c5` (origin/main, 0.3.41,
   `vectorsVersion=0.1.22`) and builds `:models-bench:installDist` with
   `--include-build <this vectors checkout>`.
   - It fails unless `vectors-core` on the Models classpath contains `GgufBandGemm`.
   - The composite build and the harness compile were verified on this laptop; the model itself
     was not run.
2. **Inputs.** Downloads or verifies the Granite 4.1 3B Q4_K_M GGUF, using the same pinned URL and
   SHA-256 `662b0626…` as the granite-alora base host run.
   - It takes the frozen window v2 from Models commit `cb2f4262` and verifies SHA-256 `dfb8cd72…`.
3. **Prefill.**
   - **Prompt:** built from the window's SQuAD document texts, then cut to the longest prefix that
     encodes to ≤ 2040 tokens (`TARGET_TOKENS`) at context 2048.
   - **Runs:** Models' own `models-bench profile-prefill` (2 warmups, then one measured prefill),
     5 fresh-JVM runs per arm, interleaved ABAB/BABA.
   - **Arms:** `-Dvectors.gguf.batchedMatmulKernel=integer` vs `=dispatch`, both with the routing
     report.
   - **Gate:** median dispatch ≥ 1.10 × median integer.
4. **Continuations.** Greedy decoding (temperature 0, max 64 tokens) of the first 20
   `squad-v2-dev` cases.
   - **Prompt:** the base-arm prompt of `models-bench activated-answerability --arm base`, rendered
     through the same public `GraniteDocumentsPrompt` calls. No adapter is loaded; base Q4_K_M only.
   - **Per case:** the token-fragment sequence, stop reason and completion token count, for each
     arm.
   - **Gate:** identical in ≥ 19/20.
   - If `ADAPTER_DIR` is set, the Models runner dumps its own base-arm prompts and the script reports
     their byte parity with the harness prompts.
5. **Report.** `compare.py` prints both results, the per-run routing proof and any observability
   problems. Everything is kept under `$WORK/evidence/<timestamp>/`.

Deviations and limits, stated before the run:

- **Continuations tool:** `activated-answerability` itself caps output at 6 tokens (hard-coded) and
  always loads an adapter. The pre-registered 64-token base-arm continuation is therefore produced by
  `ModelCheck.java` on the same Models classpath, with the same prompt renderer, not by that runner.
- **Short outputs:** the base-arm instruction asks for one word, so continuations will likely stop
  at EOS after a few tokens. The identity gate then covers the prefill-dependent first tokens, not
  64 tokens of text.
  - To cover longer text, the script also reports an `open` variant: the same cases without the
    instruction, so the model answers the question. It is reported as **supplementary and not
    gating**, because it was not pre-registered.
- **Decode arm:** decode runs at batch 1, so it is integer in both arms. Any divergence enters
  through prefill logits and KV state.
- **Prefill tool:** `models-rag-bench` was not used. The pre-registered quantities are a prefill
  rate and continuation identity; `profile-prefill` is Models' existing tool for the former.

## Model-level check, pre-registration 2 (Genoa 16 vCPU, default 256-bit species, 2026-09-17T02:28Z–03:40Z)

Measured on the reference host (`model-check/results-genoa-512/`, `summary.md`; JFR dumps kept on the
host only). Models 167a8abd composite with this branch at a475c3b; Granite 4.1 3B Q4_K_M
(sha256 662b0626…); frozen window v2 file sha256 dfb8cd72…. The routing counters confirm the
dispatch arm really took the band path (Q4_K band 39,600 / integer 240 calls; Q6_K band 5,880 /
integer 760) and the integer arm never did.

| gate (pre-registered) | integer | dispatch | result |
|---|---|---|---|
| prefill tok/s, 2,040 tokens, 5 interleaved fresh-JVM reps, median ≥ +10 % | 13.79 (12.77–14.53) | 38.86 (31.46–43.63) | **+181.8 % — pass** |
| greedy continuations, first 20 squad-v2-dev base-arm cases, identical ≥ 19/20 | — | — | **17/20 — FAIL** |
| supplementary "open" prompt (not gating) | mean 36.4 tokens | | 10/20 identical |

**Verdict: the dispatch fails its model-level gate and stays opt-in; the default does not change.**

The three base-arm divergences flip the first answer token (`Newton` → `unanswerable`,
`unanswerable` → `taxes`, `answerable` → `wave speeds`); the open variant diverges in half the
cases once continuations run long. What this does NOT establish is which arm is closer to the
model: band computes the exact float product, the integer path adds per-block activation rounding.
Two further observations: the prefill-logit checksum (sum over all ~100k vocabulary logits) moves
from 98.1 to −8,357.7, which amplifies small hidden-state differences through the summed LM-head
rows and is not by itself a correctness signal; and the dispatch runs show up to 787 ms of GC
pauses per run (the integer runs ~100–150 ms), so the band path allocates on the hot path.

Next, as a separate pre-registration written before running: both arms' greedy tokens and top-1
logit margins against an independent float reference on the same prompts (llama.cpp's F16/BF16
path or the Transformers reference already used for the Granite adapter work), to decide whether
the divergence is band error or integer error; and removing the band path's per-call allocation.

## Pre-registration 3: which arm matches the model? (written 2026-09-17T06:25Z, before any run below)

The model-level identity gate (17/20) cannot say which arm is wrong: band computes the exact float
product of the dequantised weights; the integer path adds per-block int8 activation rounding. This
experiment asks which one tracks a float reference, and is a new decision (whether the band path is
a faithful implementation), not a re-reading of pre-registration 2.

- **Prompts:** the same 20 base-arm prompts, re-rendered with `ModelCheck.prompt` on the same Models
  build; each must reproduce the recorded `promptTextSha256` in `cont-base-arm-*.json`, or the run
  stops.
- **Reference:** Transformers `ibm-granite/granite-4.1-3b` at c0650403…, weights replaced by the
  dequantised tensors of the same Q4_K_M GGUF (sha256 662b0626…; `reference_alora_case.py
  --gguf-weights` patching, q/k un-permuted), float32, no adapter, prompt tokenised without added
  special tokens, greedy, max 8 new tokens, output cut at the first end-of-text.
- **Measured per case:** reference output text, the reference's first-token top-2 logit margin, and
  exact agreement of each arm's output (from the model-check JSON) with the reference output.
- **Decision:** the band path is judged a faithful float implementation if (a) band agrees with the
  reference on at least as many of the 20 cases as integer does, and (b) on the 3 cases where the
  arms diverge, band agrees with the reference on at least 2. Otherwise the band path is suspected of
  an implementation error and is investigated before any further default decision. Either way the
  outcome is recorded; no default changes on this experiment alone.
- **Known limitation stated in advance:** the reference uses float activations, as band does, so a
  correct band implementation should agree more by construction; the experiment tests
  implementation fidelity, not which quantisation of activations is better for accuracy.

## Pre-registration 3 result: reference agreement (2026-09-17T06:40Z)

Measured on the reference host (`model-check/reference-agreement/`): all 20 prompts re-rendered with
`ModelCheck.prompt` reproduce the recorded SHA-256; reference prompt token counts equal the runtime's
on all 20; float32 Transformers with the dequantised Q4_K_M weights.

| | integer | dispatch (band) |
|---|---:|---:|
| agrees with the float reference (20 cases) | **19** | 18 |
| agrees on the 3 cases where the arms diverge | 2 | 1 |

| divergent case | reference | integer | band | reference top-2 margin |
|---|---|---|---|---:|
| 5737432b… | `Newton` | `Newton` | `unanswerable` | 0.127 |
| 5705f09e… | `unanswerable` | `unanswerable` | `taxes` | 0.381 |
| 57266193… | `wave speeds` | `answerable` | `wave speeds` | 2.000 |

**Verdict under the pre-registered rule: not established as faithful** — (a) band agreement (18)
is below integer's (19), and (b) band agrees on 1 of 3 divergent cases, below 2. Per the rule the
band path is suspected of an implementation error and is investigated before any further default
decision.

What the measurement can and cannot say: the two cases band loses are near-ties in the reference
itself (margins 0.13 and 0.38 logits); the one it wins has a margin of 2.0. Three divergent cases
cannot separate a systematic error from near-tie noise between two different float
implementations (the Java attention and norms differ from Torch's in both arms). The kernel-level
test bounds band against an exact double-precision reference at ≤ 5e-8 of the term magnitudes,
which argues against a gross kernel error. The investigation that can decide it: layer-by-layer
hidden-state comparison of both arms against the reference on these three prompts (Models already
has a per-layer observer probe), to find whether band's deviation from the reference grows at a
specific operation or stays at float-rounding level throughout.

## Pre-registration 4: where does band leave the float reference? (written 2026-09-17T06:45Z, before any code that runs)

Pre-registration 3 left the band path "not established as faithful" on three prompts whose
reference first-token margins are 0.13, 0.38 and 2.0 logits. Three token-level outcomes cannot
separate an implementation error from near-tie noise. This experiment compares hidden states layer
by layer instead. Tools: `model-check/layer-probe/` (`LayerProbe.java`, `reference_layers.py`,
`compare_layers.py`, `run-layer-probe.sh`).

**Prompts.** The three divergent base-arm prompts: `5737432bc3c5551400e51e9b`,
`5705f09e75f01819005e77a4` and `57266193dd62a815002e832e`.
- Each is re-rendered with `ModelCheck.prompt(item, ModelCheck.BASE_INSTRUCTION)` on the
  model-check Models build.
- Each must reproduce the `promptTextSha256` and `promptTokens` recorded in
  `cont-base-arm-integer.json`, or the run stops.

**Arms.** Two fresh JVMs, `-Dvectors.gguf.batchedMatmulKernel=integer` and `=dispatch`. The
routing report is recorded in each output.

**Prefill conditions.**
- **Primary: `pipeline-cache`.** Replays the model-check prefill sequence exactly. The first 20
  window cases are run in order. Each prompt rewinds to its longest shared token prefix with the
  previous prompt (as `GenerationLoop` does) and prefills only the suffix. So the batch sizes, and
  therefore the band/integer routing, are the ones the continuations saw. Decode steps are not
  replayed: they write KV only at positions past the prompt, and the next rewind discards them.
- **Supplementary: `fresh`.** A reset and one full-prompt prefill from position 0, plus an
  identical prefill with no observer installed. The logits of the two must be bit-identical, which
  shows the observer does not perturb the pass. Only `fresh` can capture every position, which the
  local-transfer analysis below needs.

**Captured stages** (last prompt position, float32), in this order:
- `embedding`: the token row times `embeddingScale`. Java recomputes this through
  `LlamaWeights.embedToken`, because the observer does not expose it. It contains no matmul, so it
  is identical in both arms; it serves as the alignment anchor.
- `layer.0` … `layer.39`: the residual stream after each decoder layer, from `LlamaForwardPass`'s
  `layerObserver`.
- `final_norm`: the forward pass's `xNorm` field, read after prefill. This is the vector the LM
  head consumed. A recomputation from the observed last layer is recorded beside it as a
  consistency check.
- `logits`: the prefill's return value, after the Granite logit scaling.

The reference side:
- **Model and weights:** Transformers `ibm-granite/granite-4.1-3b` at c0650403…, weights patched
  with `patch_with_gguf` from the same Q4_K_M GGUF, float32, `use_cache=False`, run on the Java
  arm's token ids.
- **Tokenizer check:** the prompt text tokenised with `add_special_tokens=False` must have the
  same token count; id equality is recorded.
- **Stage capture:** by module hooks — the input of layer 0, the output of each layer,
  `model.norm`, and the model's logits.
- **Index check:** `output_hidden_states=True` is also captured and cross-checked against the
  hooks, and the observed index convention is recorded. In current Transformers the last tuple
  entry is post-norm.

**Metric.** For each stage L, all relative to the reference vector's norm:

- `e_int[L] = ‖int − ref‖ / ‖ref‖`
- `e_band[L] = ‖band − ref‖ / ‖ref‖`
- `e_ib[L] = ‖int − band‖ / ‖ref‖`

**Alignment gate.** Evaluated before any verdict. `compare_layers.py` exits non-zero if any of
these fails:
- the stage lists, vector lengths, token ids or prompt SHA differ between files;
- `e[embedding] > 1e-4` in either arm;
- `e[layer.0] > 0.05`;
- `layer.0` is not closer to reference `layer.0` than to reference `embedding` and `layer.1`
  (an off-by-one check).

**Decision rule (per prompt, primary condition).**
- **Faithful:** band is judged faithful on a prompt if
  - `e_band[L] ≤ 1.25 × e_int[L]` at every stage L, and
  - `e_band[logits] ≤ e_int[logits]`.
- **Overall:** band is faithful if it is faithful on all three prompts.
- **Locating a suspect:** the first stage L* where `e_band[L*] > 1.25 × e_int[L*]` is the located
  suspect for that prompt.
- **Attention vs MLP:** the Java observer fires only after a whole decoder layer, so the two cannot
  be separated inside a Java layer's accumulated error. A supplementary `fresh` analysis narrows the
  suspect to one layer:
  - **Local transfer:** each Java arm's full-sequence output of layer L−1 is run through the
    reference layer L, with the reference's own mask and rotary inputs.
  - **Local error:** the result is compared with that arm's layer L output, giving
    `local_int[L]` and `local_band[L]`.
  - **Split:** the same transfer is also run through the reference attention half alone, which
    reports how much of the reference layer's own update is attention and how much is MLP. It
    does not split the Java arm's error.
  - Local transfer is supplementary and does not change the verdict.

**Stated in advance.**
- **Both errors are nonzero.** Both Java arms differ from Torch in attention, RMSNorm and RoPE
  implementations and in reduction order. So `e_int` and `e_band` are both nonzero, and the rule
  compares band's distance from the reference with integer's, not with zero.
- **Early-layer noise.** At early layers both errors may sit at float32 rounding level (~1e-6).
  There, a 1.25× ratio can be crossed by rounding alone. The rule is applied as written anyway. If
  L* falls where both errors are below 1e-5, the result is reported as a rule failure *with* that
  context and the next stage's ratio beside it; the rule is not relaxed after the fact.
- **Reproduction check.** If the probe's top-1 token in the primary condition does not reproduce
  that arm's recorded first fragment, the prompt is flagged as not reproducing the model-check
  condition. Its layer table is still reported.
- **What it cannot test.** The probe tests fidelity of the prefill's last position only. It says
  nothing about decode (batch 1, integer in both arms) or about accuracy. Three prompts are a
  localisation instrument, not a population estimate.
- **No default changes on this experiment alone.**

## Pre-registration 4 — probe tools (prepared here, not run here)

One command on the reference host, after `git pull` in its vectors checkout:

```bash
bash vectors-bench/jmh-results/2026-09-16-band-gemm/model-check/layer-probe/run-layer-probe.sh
# resume after a failure: OUT=/opt/layerprobe/<timestamp> bash .../run-layer-probe.sh   (FORCE=1 recomputes)
```

It uses the model-check install and harness under `/root/band-dispatch-model-check`, the evidence
directory `20260917T022838Z`, `/opt/ref/venv`, `/opt/ref/reference_alora_case.py` and the GGUF.
Window and GGUF hashes are verified. It runs four JVMs ({pipeline-cache, fresh} × {integer,
dispatch}), then the reference (with local transfer from the `fresh` dumps), then
`compare_layers.py` for each condition. Outputs go to `/opt/layerprobe/<timestamp>/`.

What was checked on this laptop (measured here; the 3B model was not run):

- **Compilation.** `LayerProbe.java` compiles against a Models 167a8abd composite built with
  `--include-build` of this branch; `vectors-core` contains `GgufBandGemm`.
- **Reflective pieces**, on Models' synthetic nano Llama GGUF (`PureJavaBackendTest.buildNanoModelFile`):
  - the forward-pass lookup, the `layerObserver` proxy install on the `private final` field, and
    the `config`, `weights`, `prefillBatchCapacity` and `xNorm` reads;
  - the all-positions dump size.
  - **Paths covered:** F32 projections (token-at-a-time path), and Q4_0 at prefill batch 32 and 5
    (batched and chunked path). Under the `integer` and `dispatch` properties, these pass:
    - the observer reports every layer;
    - `xNorm` equals `rmsNorm` of the observed last layer, bit-for-bit;
    - a suffix prefill after `rewind` reports only suffix positions, and its logits match a
      fresh prefill.
- **Analysis scripts, end to end**, on a tiny random `GraniteForCausalLM`, with fake "Java" files
  made from its own hidden states plus noise:
  - `reference_layers.py` and `compare_layers.py`, under transformers 4.46.3 / torch 2.2.2 (the
    newest torch for this Intel Mac). The host has transformers 5.17 / torch 2.14, which was not
    exercised here.
  - Hook stages chain exactly, and the `output_hidden_states` tuple matches the hooks. Its last
    entry is post-norm (`hs_last_is=final_norm`).
  - The local-transfer self-replay reproduces the reference layer output exactly (max|diff| 0.0).
- **Rule and alignment tests.** `test_compare_layers.py` has 15 synthetic tests:
  - the off-by-one, embedding, stage-list, length, token-id and kernel-label misalignments, each
    exiting 2;
  - the inclusive 1.25 boundary, L* location, the logits clause, and a zero-integer-error stage.

Facts about the pass that shape the probe (read in Models 167a8abd source):

- **Prefill chunking.** The default prefill batch capacity is 32 (`PureJavaPlanConfiguration`).
  A ~300-token prompt is prefilled in chunks of 32, and the last position sits in the remainder
  chunk; a remainder of 1 runs the single-token path. So band routing at the last position
  (Q4_K band at ≥ 4, Q6_K at ≥ 32) depends on the prefill length. The prefill length differs
  between `fresh` and the model-check's prompt-cache reuse, which is why `pipeline-cache` is
  primary. Each output records `lastPositionChunkSize`.
- **Observer side effects.** Installing an observer disables only the final-layer
  pruning/KV-only shortcuts. Those are rejected for Granite anyway (`usesStandardLlamaLayerSemantics`
  is false), so for Granite the observer should not change the executed path. The `fresh` control
  (bit-identical logits) measures this on the host.
- **Embedding.** The observer does not expose the embedding. It is recomputed from
  `LlamaWeights.embedToken × embeddingScale`, which is the same code the pass runs, but it is not an
  observation of the pass.

## Pre-registration 4 result: alignment gate FAILED, so no verdict (run 2026-09-17T06:56Z–07:05Z)

Host: the reference host (16 vCPU, 30 GiB, x86_64; `results-20260917T065647Z/host.txt`). Both conditions ran to
completion. Every Java run reproduced its recorded first fragment. Routing was confirmed in both
arms, the observer was bit-identical, and the Hugging Face token ids equalled the Java ids.

**Gate outcome.** `compare_layers.py` exited 2 in both conditions, and on all three prompts it
failed the same check: the **integer** arm's `e[layer.0]` exceeds the pre-registered absolute bound
of 0.05 (0.0585, 0.0644, 0.0591). The band arm passes that bound (0.041–0.047). Every other gate
check passed:

- `e[embedding]` is 0 in both arms.
- `layer.0` is about 10× closer to reference `layer.0` than to reference `layer.1` (0.66–0.69), and about 20× closer than to the embedding (1.12–1.14). There is no off-by-one.

The 0.05 bound was a guess written before any data. Under the rules stated above, **no faithful or
unfaithful verdict is issued**, and the bound is not relaxed after the fact. A rerun with a revised
gate must be pre-registered first. Local transfer did not run, because it sits after the gate.

**Exploratory observation (not a verdict).** `layer_observation.py` computed the same `errors()`
rows without the gate (`results-20260917T065647Z/layer-observation.json`). Measured here:

| condition | prompt | stages where e_band > e_int | e_int / e_band at layer.0 | at layer.20 | at logits |
|---|---|---|---|---|---|
| pipeline-cache | 5737432b | 0 of 43 | 0.0585 / 0.0457 | 0.0555 / 0.0239 | 0.0872 / 0.0260 |
| pipeline-cache | 5705f09e | 0 of 43 | 0.0644 / 0.0437 | 0.0588 / 0.0226 | 0.1206 / 0.0293 |
| pipeline-cache | 57266193 | 0 of 43 | 0.0591 / 0.0415 | 0.0607 / 0.0248 | 0.1132 / 0.0277 |
| fresh | 5737432b | 0 of 43 | 0.0585 / 0.0472 | 0.0555 / 0.0254 | 0.0872 / 0.0263 |
| fresh | 5705f09e | 0 of 43 | 0.0644 / 0.0447 | 0.0588 / 0.0224 | 0.1206 / 0.0519 |
| fresh | 57266193 | 0 of 43 | 0.0591 / 0.0415 | 0.0607 / 0.0251 | 0.1132 / 0.0285 |

What this suggests, labelled as belief until a pre-registered rerun confirms it:

- The integer path's int8 activation quantisation, not the band path, is the larger departure from the float reference. At the logits, band sits 3–4× closer on every prompt.
- On these three near-tie prompts, the smaller hidden-state error did not buy top-1 agreement. The reference, integer and band top-1 tokens were Newton/Newton/un, un/un/tax and wave/answer/wave. Prompts whose top-2 margin is 0.13–2.0 logits flip on errors of this size, whichever arm is closer.
- Band sitting farther from integer (`e_ib`) than from the reference at late layers is consistent with that reading.

**Next, if pursued:** pre-register a gate whose layer.0 bound is relative (for example, the arm's
`layer.0` error well below its distance to the neighbouring reference stages). Then rerun the
unchanged decision rule and local transfer. Also pre-register a population measure: logit error
against the reference over all 20 prompts, rather than top-1 on three near-ties. No default changes.
