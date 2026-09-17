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
