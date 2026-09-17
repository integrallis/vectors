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
