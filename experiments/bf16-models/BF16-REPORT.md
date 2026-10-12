# BF16 and executor qualification

**Status: final whole-model qualification is running; release acceptance is not yet established.**

The corrected BF16 kernel is deterministic across compilation and sequential/batched execution. Every measured kernel latency interval excludes a regression on x86, SVE ARM and NEON ARM. Earlier complete model runs exposed regressions outside the BF16 kernel; their failures are retained, and the corrected runtime is being measured with fresh samples.

## Frozen artifacts and scope

| Evidence | Runtime | Scope |
|---|---|---|
| Final macro campaign v9/v10/v11 | Vectors `ae337eeb`; Models `0bc5de31` | Fresh real-weight fixtures and ten same-host triplets per model/architecture |
| Executor maximum-range correction | Vectors production `a40bcb71` | Adds overflow-safe range arithmetic after macro freeze; separate actual-artifact correctness proof |
| Earlier complete Qwen model campaign | Vectors `d92aceb0`; Models `12796ebf` | Historical regression evidence; never pooled with final measurements |

Frozen macro core JAR SHA256: `696c2a58843181a2b02c91043208b56c9c6ebdc976e0d3f44dc051cc1f36d555`.
The BF16 class is byte-identical across the kernel and successive macro artifacts: `93d0a1f0022290ac0dcc008b3270e50c46c174b6c4cd01a307614d3df80bc80d`.

## Changes and correctness

- BF16 accumulation now follows one reduction tree. Two input rows reuse decoded weights. Large shareable matvecs use the existing persistent pool; SVE selects its measured inline reduction layout with identical arithmetic. No additional tuning flag or pool is introduced.
- Nested callbacks run inline across pools, preventing same-pool and A→B→A deadlocks. Closed pools reject nested execution, and closing pools from callbacks fails before a barrier cycle can form.
- Maximum row and chunk counts use overflow-safe ceiling/range arithmetic. The old artifact emits negative rows, drops all stage work, or hangs. Four bounded actual-artifact probes pass after correction; 70 normal partition configurations retain identical old/new hashes. Twenty-one focused executor tests pass. See [executor proof](../executor-ranges/PROOF.md).
- Exact activation retains the stable arithmetic implementation. An array boundary prevents temporary vector objects escaping when SwiGLU is compiled in a full model; see [actual-model allocation evidence](../activation-boundary/README.md).
- Qwen BF16 and Bonsai real-weight fixtures pass pinned reference outputs, cloned full-logit repeat/reset checks, and chunk-boundary sequential/batch equality on x86, SVE ARM and NEON. All six final v9/v10/v11 hosts repeat these gates successfully: 24 candidate/control checks pass, while all 12 published-arm checks reproduce the known determinism failure. See the [complete actual-artifact fixture manifest](final-real-fixtures/manifest.json). The published stack fails fresh-session determinism; its failures remain recorded.
- The Bonsai harness uses the CI prompt, context 96, batch 1/batch 16, native default route and three complete vocabulary snapshots. Its fixture uses explicit sessions; the separate unchanged literal JUnit method using implicit overloads passes against the overflow-fixed artifact, with zero full-vocabulary absolute/scaled error at all three continuations. [Literal test evidence](literal-bonsai-ci/provenance.json) pins the actual JARs, native binary and unchanged test source.

## Complete BF16 kernel measurements

Each row uses all twelve counterbalanced independent process pairs, geometric mean latency ratios and a paired percentile bootstrap (20,000 resamples, seed 716309). Ratio below one is faster. No trial was removed. Precision is a separate declared two-percentage-point interval-width target; a miss remains a miss.

| Host | Operation | Shape | Ratio | 95% interval | Precision target | Bytes/op candidate / published |
|---|---|---|---:|---|---|---:|
| x86 | MatVec | 151936x896 | 0.290252 | [0.289425, 0.291132] | met | 93.222 / 180.509 |
| x86 | MatVec | 4864x896 | 0.345798 | [0.323948, 0.358173] | not met | 41.423 / 233476.213 |
| x86 | MatVec | 48x5120 | 0.765070 | [0.750964, 0.783497] | not met | 2304.196 / 2304.257 |
| x86 | Prompt batch | 151936x896 | 0.844521 | [0.840239, 0.849300] | met | 532.300 / 602.201 |
| x86 | Prompt batch | 4864x896 | 0.847929 | [0.838234, 0.857633] | met | 63.653 / 66.462 |
| x86 | Prompt batch | 48x5120 | 0.594734 | [0.568733, 0.610359] | not met | 4.074 / 6.863 |
| SVE ARM | MatVec | 151936x896 | 0.360530 | [0.359651, 0.361458] | met | 87.450 / 130.820 |
| SVE ARM | MatVec | 4864x896 | 0.459692 | [0.454664, 0.464637] | met | 41.543 / 233475.421 |
| SVE ARM | MatVec | 48x5120 | 0.972874 | [0.969858, 0.976104] | met | 0.133 / 2304.139 |
| SVE ARM | Prompt batch | 151936x896 | 0.947059 | [0.925442, 0.967655] | not met | 503.718 / 525.712 |
| SVE ARM | Prompt batch | 4864x896 | 0.931279 | [0.917247, 0.945795] | not met | 61.945 / 62.898 |
| SVE ARM | Prompt batch | 48x5120 | 0.951004 | [0.950036, 0.951761] | met | 2.879 / 51.027 |
| NEON ARM | MatVec | 151936x896 | 0.255304 | [0.254958, 0.255635] | met | 261.417 / 857.600 |
| NEON ARM | MatVec | 4864x896 | 0.527202 | [0.525646, 0.528760] | met | 47.214 / 175117.836 |
| NEON ARM | MatVec | 48x5120 | 0.982108 | [0.979721, 0.985095] | met | 1728.684 / 1728.697 |
| NEON ARM | Prompt batch | 151936x896 | 0.983110 | [0.980846, 0.985405] | met | 2630.400 / 2631.200 |
| NEON ARM | Prompt batch | 4864x896 | 0.985702 | [0.982769, 0.988738] | met | 138.245 / 139.886 |
| NEON ARM | Prompt batch | 48x5120 | 0.973483 | [0.973195, 0.973744] | met | 67.897 / 68.432 |

All kernel cold/hot, supported forced-width, allocation and nested-dispatch gates passed before timing. NEON ran on an actual m6g host with `sve=false`, preferred 128; forced widths on SVE are not substituted for this hardware measurement. No NEON whole-model performance claim is made.

## Whole-model procedure and remaining evidence

The final v9/v10/v11 payloads contain the same immutable Models/Vectors artifacts. Pair blocks 0–3, 4–6, 7–9 stay on their respective hosts; each published/control/candidate triplet remains on one host in its predeclared counterbalanced order. Every process performs one cold request, three warmups and three measured requests. The ten blocks cover 30 distinct measured tasks from JEVBench and SNIPS. Bootstrap resampling preserves each host stratum and its observed block count. Confidence intervals crossing one are unresolved; faster total requests cannot erase a regressed prefill, allocation or startup gate.

The campaign deadline is 22:32 UTC; per-phase caps and the approved combined $12 continuation budget remain enforced. Only complete triplets are aggregated. Interrupted or partial results remain raw evidence and cannot become a passing result. The separate release campaign owns Bonsai RAG and 200 public-QA qualification.

Historical d92 Qwen completed all ten triplets on each host: candidate/control logits, tokens and predictions match exactly. All three arms scored 10/15 JEVBench and 14/15 SNIPS; this small classification sample is not general model-quality certification. Candidate/control prefill improves on both hosts, but candidate/published ARM prefill regresses 1.027957 [1.017914,1.040664]. Early Bonsai d92 allocation was 2.3–2.5 times published. These are rejected historical results, not accepted tradeoffs. Fresh ae337 samples must establish whether the array-boundary fix resolves them. The first final ARM Bonsai triplet still allocates 1.484 GB per measured candidate request versus 610 MB published (corrected control 1.485 GB); this early complete block exposes a remaining defect and has triggered separate JFR diagnosis. It is not a ten-pair statistical result. Full historical metrics remain in [the retained report](HISTORICAL-D92-REPORT.md) and per-host JSON.

## Final ARM Qwen BF16 result

All ten final process triplets completed across the three declared hosts. The corrected control and candidate match exactly for all 70 complete request snapshots, tokens and predictions. All three arms score 10/15 JEVBench and 14/15 SNIPS. These are limited qualification samples.

| Metric | Candidate/control ratio [95% CI] | Candidate/published ratio [95% CI] |
|---|---|---|
| loadNanos | 0.982700 [0.963009, 1.002823] (unresolved) | 0.991161 [0.968828, 1.012460] (unresolved) |
| coldRequestNanos | 0.793877 [0.785496, 0.802098] (faster) | 0.995933 [0.985281, 1.007229] (unresolved) |
| prefillNanos | 0.931223 [0.924654, 0.937283] (faster) | 1.009760 [1.000627, 1.017865] (slower) |
| decodeNanos | 0.236024 [0.226378, 0.247790] (faster) | 0.226042 [0.216452, 0.237540] (faster) |
| requestNanos | 0.670679 [0.657724, 0.683766] (faster) | 0.692684 [0.678091, 0.706792] (faster) |
| peakRssKiB | 0.838715 [0.833594, 0.842969] (faster) | 0.845979 [0.839776, 0.851429] (faster) |
| javaAllocatedBytes | 0.067532 [0.065874, 0.069664] (faster) | 0.088061 [0.085949, 0.090773] (faster) |

**The final candidate still fails the published-stack ARM prefill gate.** Its 1.009760 ratio has interval [1.000627, 1.017865]; faster decode and request totals do not remove this regression. Startup and cold-request intervals cross one and remain unresolved. Allocation improves substantially. Every metric retains its separate precision-target result in [the analysis](combined-final-arm-qwen-bf16-analysis.json); the complete paired raw files and artifact/host provenance are in [final-results/arm](final-results/arm/provenance.json). Remaining shared-runtime prefill work must be diagnosed before whole-model acceptance.

## Build, harness and provenance

The preceding ARM core CI ran 1,601 cases with 1,597 passes and four hardware skips. Two local forced 64-bit child timeouts under severe contention passed unchanged 180-second retries; the original failures remain retained. Frozen ae337 release dry-run 38307108275 passed. Overflow-final head a40bcb71 CI and release dry-run 38310390999 are running.

The original macro helper retained backend-owned logit arrays. Those per-step hashes and quality labels are invalid. The model-fixture helper always cloned immediately and remains valid. Corrected macros snapshot immediately, verify ownership with a deliberately reused-buffer probe, and separately record copy time/allocation before subtracting only that measured overhead from request totals. Raw totals remain available. All arms use the corrected harness.

`FINAL-PROTOCOL.md`, `collect-final.py` and per-phase payload/host receipts define the final analysis. The collector verifies immutable payload hashes and actual JAR hashes before combining complete host-paired blocks; old/new timings are never pooled. Original failed reduction/layout experiments are preserved by merging their source history. No final release acceptance is claimed until all required gates are resolved.
