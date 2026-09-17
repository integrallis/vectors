# 2026-09-17 — Vector API K-quant dequantisation for the band GEMM arm

Branch `exp/kquant-dequant-simd`, cut from `exp/band-gemm` at `a475c3b`. This section (question,
arms, decision rule) is committed **before any run**. Everything below the `---` line was added
afterwards.

## Question

The band GEMM arm (`GgufBandGemm`, `2026-09-16-band-gemm/README.md`) dequantises the whole weight
matrix to F32 on every call, whatever the batch. On Genoa its scalar dequant ran at 1.3 G F32
elements/s for Q4_K and 0.6 G for Q6_K (single thread, measured there), and the band arm was
0.27x–0.39x as fast as the integer kernels at batch 1, single-threaded, on the Granite FFN shapes.
The working hypothesis (believed, from the component ablation, not proven) is that dequant is the
dominant decode cost.

Does a Panama Vector API dequantisation of Q4_K and Q6_K super-blocks (bit-identical to the scalar
one) make the band arm faster at small batches, without costing prefill?

Q5_K is out of scope: the band arm does not support it. Q8_0 dequant is not changed (it already runs
at 3.7 G elements/s on Genoa).

## Arms (selected with `-Dvectors.gguf.band.dequant=…`, resolved once per JVM)

| name | what it does |
|---|---|
| `scalar` | the current per-block scalar loops (`dequantizeQ4_KBlock`, `dequantizeQ6_KBlock`); stays as the reference path and the default until this rule is decided |
| `simd-byte` | fused: load `L` packed bytes (`L` = float lanes), extract nibbles / 2-bit high parts with **byte-lane** `AND`/`LSHR`, `B2F`, multiply by the sub-block scale and subtract the sub-block min in `FloatVector` lanes |
| `simd-int` | fused: load `L` bytes, **`B2I` first**, extract in **int lanes**, `I2F`, then the same float arithmetic (tests the claim, read in qxotic's comments and not measured here, that int lanes beat byte lanes for nibble unpack) |
| `simd-split` | two stages: unpack a whole sub-block group into signed quant bytes with wide byte lanes (`ByteVector` 256 or 128), then convert and scale in `L` float lanes, or in a scalar loop where no byte species with `L` lanes exists |

The fused arms need a byte species with exactly `L` lanes (8 → 64-bit, 16 → 128-bit). There is no
4-lane byte species, so at 128-bit float species (`L` = 4) a fused request runs `simd-split`; the
effective arm is reported, never silent. Every arm reports `requested`, `effective` and `lanes` in
`GgufBatchedMatmulKernel.bandDequantConfiguration()`, and every benchmark fork prints it.

## Decision rule (recorded before running)

A SIMD arm **replaces the scalar dequant as the band arm's default** only if all of:

- **(a) Bit-identity.** Every SIMD arm's output equals `scalar` bit for bit
  (`Float.floatToIntBits`, i.e. NaN payloads canonicalised) for Q4_K and Q6_K on randomized blocks
  and on adversarial blocks — all-zero bytes, all-0xFF bytes, maximum 6-bit scales and mins (63),
  maximum and minimum int8 Q6_K scales (127, −128), negative, subnormal, ±max-finite and ±0 float16
  `d`/`dmin` (negative `dmin` makes the subtracted min negative), and non-finite float16 — in-process
  and in fresh JVMs at 128-bit and 256-bit species (and 512-bit where the host has it). The existing
  band GEMM, dispatch and structural tests pass unchanged.
- **(b) Throughput.** The candidate's dequant-only throughput (`GgufBandDequantBenchmark`, single
  thread) is **≥ 2.0x** the `scalar` arm's for both Q4_K and Q6_K, on both Granite FFN shapes, with
  256-bit species, on at least one host.
- **(c) End-to-end.** On the Genoa runs (both `MAXBITS=512` and default 256-bit species),
  `GgufBandGemmAbBenchmark` with `kernel=BAND_F32`, candidate vs `scalar`, for Q4_K and Q6_K on
  `8192x2560` and `2560x8192`, multi-threaded (the executor default):
  - at **batch 1 and batch 4**, the candidate's mean ms/op is lower than `scalar`'s in every cell
    (8 cells per run);
  - at **batch 512**, the candidate's mean ms/op is at most **1.03x** `scalar`'s in every cell.

**The candidate** is fixed as the SIMD arm with the highest geometric-mean dequant throughput over
{Q4_K, Q6_K} × {8192x2560, 2560x8192} on the Genoa 256-bit run. (a) must hold for all arms; (b) and
(c) are evaluated for the candidate only.

**Which evidence decides.** The i7-9750H laptop used for development is heavily shared: its numbers
are a screen, labelled as such, and cannot decide (c). (b) can be met there only if the effect
repeats across at least two local runs; otherwise it waits for Genoa.

**Not decided here.** The dispatch thresholds (Q4_K 4, Q6_K 32, Q8_0 4) and the dispatch mode's
behaviour are not changed on this branch. A faster dequant may move the crossovers; a separate,
later rule may revisit them with a new crossover sweep. Batch 2 and 8 are measured for that purpose
and do not enter this rule.

If the rule fails, the SIMD arms stay selectable experiments and `scalar` stays the default.

---

## Implementation (`vectors-core`, `GgufKQuantDequant`)

- Three SIMD arms as pre-registered (`simd-byte`, `simd-int`, `simd-split`), plus `scalar`, which
  calls the unchanged `GgufBandGemm.dequantizeQ4_KBlock` / `dequantizeQ6_KBlock` reference loops.
  Every SIMD loop keeps a scalar tail. At 8 and 16 float lanes no tail runs, because the lane count
  divides the 16- and 32-element groups.
- **Species:** all `static final`. `LANE_BYTE_SPECIES` is 64-bit at 8 lanes and 128-bit at 16.
  `LANE_INT_SPECIES` is 256-bit or 512-bit. `WIDE_BYTE_SPECIES` is 256-bit, or 128-bit when the float
  species is 128-bit. The structural test now also has to reach `GgufKQuantDequant`.
- **Scales:** `d`, `dmin` and the 6-bit or int8 sub-block scales are decoded once per sub-block into
  the same scalar `float` expressions as the reference. Only `q` extraction and the lane-wise
  `scale * q - min` are vectorised, which is why the outputs are bit-identical.
- **Selection:** `-Dvectors.gguf.band.dequant` (default `scalar`). The band arm's `Plan` carries the
  arm, and `GgufBandGemm.gemm/dequantize` have explicit-arm overloads that the tests and the dequant
  benchmark use.
- **Reporting:** `GgufBatchedMatmulKernel.bandDequantConfiguration()` reports it as
  `kquant-dequant(requested=…,effective=…,lanes=…)`. `describe()`, the capabilities string, dispatch
  thresholds and dispatch behaviour are **unchanged**.
- **Test-only call counters** (`-Dvectors.gguf.band.dequant.count=true`) show which arm the band
  path actually ran. The flag is a static final boolean, so it folds away when off.

## Tests (measured here: red → green)

- **Red:** `GgufKQuantDequantTest` and `GgufKQuantDequantProbe` were committed first (`f7898da`) and
  did not compile without the implementation (108 compiler errors).
- **Green:** after `d1feecd`, results were `GgufKQuantDequantTest` 27/27, `GgufBandGemmTest` 32/32
  (unchanged), `GgufBatchDispatchTest` 8/8 (unchanged) and `VectorSpeciesConstantStructureTest` 4/4.
- **Bit-identity** is checked with `Float.floatToIntBits` for every arm × {Q4_K, Q6_K} on these
  blocks:
  - all-zero and all-0xFF blocks;
  - a 13×13 grid of edge float16 `d`×`dmin` values: ±0, ± smallest subnormal, largest subnormal,
    smallest normal, ±1, ±65504, ±inf, NaN (negative `dmin` gives negative mins);
  - 6-bit scales and mins at 63 in both packings, and scales-only / mins-only packings;
  - Q6_K int8 scales at 127, −128, 0, 1 and −1, with 0x00/0xFF/0xAA/0x55/random quant bytes;
  - alternating 127/−128 sub-block scales;
  - 400 random blocks, half fully random and half with realistic `d`.
  Multi-block runs also check the source offset and a non-zero destination offset.
- **Fresh-JVM probe** at `maxBits` 128 and 256 (512 is added automatically on AVX-512 hosts), for each
  of the 4 arms:
  - the reported requested and effective arm (fused → split at 4 lanes);
  - the call counters show that only the effective arm ran;
  - band matmul output on 4 shapes, including the parallel `5x512x2560` one, is bit-identical to
    band matmul with the `scalar` dequant;
  - 40 repeated block passes, so C2-compiled kernels are compared, not only the interpreter.
- **Mutation checks (measured):** each mutant failed exactly its own arm × format cell in both
  in-process bit-identity tests, and nothing else. The mutants were:
  - `simd-byte` Q4_K high-nibble shift 4→3;
  - `simd-int` Q4_K mask 0x0F→0x1F;
  - `simd-split` Q4_K dropping `- min`;
  - `simd-byte` Q6_K bias 32→31;
  - `simd-int` Q6_K high-bit shift 4→5;
  - `simd-split` Q6_K reading `qh` from `ql`.

## Local screen — loaded laptop, NOT a decision (measured here)

Intel i7-9750H, AVX2 + FMA, 256-bit species (8 lanes), Temurin 25.0.3, macOS.

- **Load:** the host was shared with other sessions. The 1-minute load average was **62–476** during
  the multi-threaded screen and **37–72** during the single-threaded screen (`raw/*/load.txt`).
- **JMH:** 1 fork, 2×1 s warmup, 3×1 s measurement. The 99.9 % error bars are often larger than the
  scores. Read only directions that repeat.
- **Build:** the multi-threaded screen ran on the jar built at `d1feecd` plus the then-uncommitted
  benchmark edits committed as `cb069be` (`dirty=2` in `host.txt`). The single-threaded screen ran
  at `cb069be` (clean).

### Dequantisation alone (single thread), two runs

| format | shape | run 1 (`dequant-prerun-xmx4g.json`): byte / int / split vs scalar | run 2 (`dequant.json`): byte / int / split vs scalar |
|---|---|---|---|
| Q4_K | 8192x2560 | 3.26x / 4.10x / 3.99x (scalar 50.0 ms) | 2.52x / 3.86x / 6.01x (scalar 126 ms, ± 846) |
| Q4_K | 2560x8192 | 3.07x / 3.84x / 3.81x (scalar 48.3 ms) | 7.97x / 6.90x / 6.06x |
| Q6_K | 8192x2560 | 2.73x / 4.51x / 3.93x (scalar 86.2 ms) | 5.17x / 8.21x / 5.70x |
| Q6_K | 2560x8192 | 3.74x / 6.86x / 5.50x (scalar 126 ms) | 5.25x / 4.98x / 7.15x |

- **Repeats:** every SIMD arm was ≥ 2x scalar in every cell in both runs. Run 1 was the quieter run
  (scalar Q4_K at 48–50 ms ≈ 430 M elements/s).
- **Run 1 throughput:** `simd-int` ≈ 1.7 G elements/s on Q4_K and 1.1–1.4 G on Q6_K; `simd-split` was
  close behind.
- **Byte vs int lanes:** `simd-byte` was the slowest SIMD arm in run 1, consistent with the claim that
  int lanes beat byte lanes for nibble extraction, on AVX2, where byte shifts have no native
  instruction. Run 2 is too noisy to rank the arms.
- **Rule (b):** under the pre-registered wording, (b) is **met on this host** (≥ 2.0x, 256-bit
  species, repeated in two runs) for all three arms. The candidate is still fixed by the Genoa
  256-bit run.

### Band arm end-to-end, multi-threaded (12 executor threads), two rounds

Full tables: `summary-intel-i7-9750h-avx2-screen.md`.

- **Batch 1 and 4 (new/old = scalar band ms ÷ SIMD band ms):**
  - `simd-split`: 1.03x–3.30x in round 1 (8/8 faster); 0.87x–4.35x in round 2 (7/8; Q4_K
    `2560x8192` batch 4 was 0.87x).
  - `simd-int`: 1.17x–2.25x in round 1; 0.61x–3.30x in round 2.
  - `simd-byte`: 0.66x–3.06x, the least consistent.
- **Batch 512:** `simd-split` was 1.15x–1.97x (round 1) and 1.10x–1.38x (round 2), never slower.
  `simd-int` and `simd-byte` had batch-512 cells at 0.59x–0.89x. The error bars span both
  directions, so this cannot say whether the fused arms really lose at prefill.
- **Integer arm:** at batch 1 the band arm still did not beat it reliably with multiple threads
  (`simd-split` vs integer 0.55x–1.8x across rounds).

### Band arm end-to-end, single-threaded (`-Dvectors.gguf.parallel=false`), one round, load 37–72

| format | shape | batch | integer ms | band scalar ms | band simd-int ms | band simd-split ms | int new/old | split new/old | split vs integer |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4_K | 2560x8192 | 1 | 37.3 | 60.7 | 19.1 | 14.2 | 3.18x | 4.28x | 2.63x |
| Q4_K | 2560x8192 | 4 | 108.8 | 55.3 | 23.0 | 33.8 | 2.41x | 1.64x | 3.22x |
| Q4_K | 8192x2560 | 1 | 38.4 | 53.3 | 17.4 | 15.3 | 3.07x | 3.48x | 2.50x |
| Q4_K | 8192x2560 | 4 | 132.5 | 49.1 | 21.6 | 37.5 | 2.27x | 1.31x | 3.53x |
| Q6_K | 2560x8192 | 1 | 56.7 | 111.5 | 25.0 | 28.7 | 4.47x | 3.89x | 1.98x |
| Q6_K | 2560x8192 | 4 | 101.9 | 105.2 | 50.8 | 34.7 | 2.07x | 3.03x | 2.94x |
| Q6_K | 8192x2560 | 1 | 61.6 | 150.6 | 24.5 | 34.2 | 6.14x | 4.40x | 1.80x |
| Q6_K | 8192x2560 | 4 | 107.9 | 113.1 | 29.3 | 38.1 | 3.86x | 2.97x | 2.84x |

Full table with error bars: `summary-intel-i7-9750h-avx2-screen-st.md`.

- **Every cell** favours both SIMD arms over scalar dequant.
- **Against integer at batch 1:** the band arm with SIMD dequant went from **0.41x–0.72x** to
  **1.8x–2.6x**. This is a single round on a loaded host, so it is a **hypothesis for Genoa, not a
  result**. It suggests dequant was indeed the dominant single-thread decode cost here.

### Local verdict (screen only)

- **(a) met:** bit-identity for all arms at 128- and 256-bit species; the existing tests are
  unchanged and green.
- **(b) met on this host** for all three arms, in two runs.
- **(c) not decidable here:** by the rule's own terms it needs Genoa. The laptop direction favours
  `simd-split` (batch 1/4 faster in 15 of 16 cells over two rounds, and never slower at batch 512).
  The fused arms' batch-512 cells are unresolved.
- **Default:** stays `scalar` until the Genoa runs decide.
- **Not measured:** NEON, and any 128-bit-species performance (correctness only). 512-bit species
  were not exercised here (no AVX-512).

## Genoa runs (to be run by the coordinator; nothing below is measured yet)

```bash
git clone git@github.com:integrallis/vectors.git && cd vectors && git checkout exp/kquant-dequant-simd
./gradlew :vectors-bench:jmhJar
R=vectors-bench/jmh-results/2026-09-17-kquant-dequant
MAXBITS=512 $R/run.sh "$(hostname)-avx512-512bit" dequant ab      # 512-bit species, 16 lanes
$R/run.sh "$(hostname)-256-on-avx512" dequant ab                  # library default 256-bit, 8 lanes
python3 $R/summarize.py $R/raw/"$(hostname)-256-on-avx512"        # candidate = best geomean here
python3 $R/summarize.py $R/raw/"$(hostname)-avx512-512bit" --candidate <that arm>
```

- **Defaults:** 2 forks, 3×2 s warmup, 5×2 s measurement. Each species run has two phases:
  - `dequant`: 16 cells;
  - `ab`: integer plus 4 band arms × 20 cells.
  That is roughly 70 minutes per species run on a 16-vCPU host (believed, from cell counts; not
  timed).
- **Optional:**
  - `ROUNDS=2` repeats `ab` with the arm order rotated;
  - adding the `ab-st` phase gives the single-thread table above (not part of the rule);
  - `THREADS=<physical cores>` stops the executor from oversubscribing SMT siblings.
- **Guard:** each band fork fails unless it runs the labelled arm, and prints `bandDequant=…` in
  `raw/<label>/ab-band-<arm>.txt`.
