# BF16 production validation

Status: qualification is incomplete. Qwen ARM prefill regresses against the published stack;
load-time confidence intervals remain unresolved. Bonsai whole-model qualification is running. The complete twelve-pair kernel matrix
excludes a latency regression for every measured x86 and SVE ARM shape. Several confidence
intervals exceed the desired two-percentage-point width; those precision misses remain visible.
Non-SVE ARM is being measured on a separate m6g host.

The frozen final runtime is `d92aceb0cabde1436098c7461f95b94e6f09d9cb`; its core jar SHA256 is
`3b037e5e405b526e67cd4078c58e191e60589b7b3dbccb9992f8d98d258f5202`.
Kernel timing used predecessor `dd58fdefd972b4b6709d0613a9aa725a96ee8b82`. The BF16 class is
byte-identical in both artifacts (`93d0a1f0022290ac0dcc008b3270e50c46c174b6c4cd01a307614d3df80bc80d`).
The final runtime additionally rejects nested execution on closed pools; fresh real-model fixtures
have passed on both hosts, and macros use that artifact. Neither artifact contains a rejected activation
prototype: both use the stable exact activation implementation.

## Kernel measurements

Numbers below are candidate/published latency ratios; smaller is faster. Each comparison uses all
twelve counterbalanced independent process pairs, geometric mean ratios and a paired percentile
bootstrap (20,000 resamples, seed716309). No trial was removed.

| Host | Operation | Shape | Ratio |95% interval|2-point precision target|Bytes/op candidate / published|
|---|---|---|---:|---|---|---:|
|x86|MatVec|151936x896|0.290252|[0.289425,0.291132]|met|93.222 / 180.509|
|x86|MatVec|4864x896|0.345798|[0.323948,0.358173]|not met|41.423 / 233476.213|
|x86|MatVec|48x5120|0.765070|[0.750964,0.783497]|not met|2304.196 / 2304.257|
|x86|PromptBatch|151936x896|0.844521|[0.840239,0.849300]|met|532.300 / 602.201|
|x86|PromptBatch|4864x896|0.847929|[0.838234,0.857633]|met|63.653 / 66.462|
|x86|PromptBatch|48x5120|0.594734|[0.568733,0.610359]|not met|4.074 / 6.863|
|arm|MatVec|151936x896|0.360530|[0.359651,0.361458]|met|87.450 / 130.820|
|arm|MatVec|4864x896|0.459692|[0.454664,0.464637]|met|41.543 / 233475.421|
|arm|MatVec|48x5120|0.972874|[0.969858,0.976104]|met|0.133 / 2304.139|
|arm|PromptBatch|151936x896|0.947059|[0.925442,0.967655]|not met|503.718 / 525.712|
|arm|PromptBatch|4864x896|0.931279|[0.917247,0.945795]|not met|61.945 / 62.898|
|arm|PromptBatch|48x5120|0.951004|[0.950036,0.951761]|met|2.879 / 51.027|

The previously unresolved ARM small matvec is now resolved by the prospective SVE layout policy.
All twelve cold/hot width/shape checks, standalone allocation checks and nested executor checks
passed before timing on both hosts. Hardware branch output and raw receipts accompany the results.

## Correctness and lifecycle

- The real Qwen BF16 and Bonsai fixtures passed on both original final hosts: pinned reference
  logits/tokens, repeated fresh-session full-logit hashes, implicit reset and exact sequential versus
  batched prefill across a chunk boundary. The published stack failed fresh-session hash equality for
  both models on both hosts. Those failures are retained, not ignored or turned into acceptance.
- Final source's ARM CI core suite has1,601 cases:1,597 passed, four existing256-bit GGUF dequant
  cases were hardware-skipped, and none failed. The CI merge tree equals the final branch tree.
- The final local full run had1,598 passes and two forced64-bit child-process timeouts under severe
  concurrent load. Both timeout cases passed on the final artifact under the unchanged180-second
  bound (35.45s and97.44s). Original failures and the active computation thread dump are retained.
- Same-pool and A→B→A nested dispatch regressions have red/green child-process evidence. Closed
  nested executors now reject work;18 focused executor tests and the expanded child probe pass.
- The final local Qwen model fixture also passes on the final jar.

## Harness correction and outstanding gates

The model-fixture helper always cloned each logit buffer before the next inference call; its fixture
results remain valid. The original macro helper retained backend-owned arrays. Its per-step hashes
and first-token classification scores cannot be used. That macro phase was superseded after the
complete kernel matrix, and both original hosts were confirmed terminated.

The corrected macro helper clones immediately, verifies ownership with a deliberately reused-buffer
probe, records capture overhead separately and subtracts only measured copy time/allocation from
observed request totals. Raw totals are retained too. All three arms use the same corrected harness.
Bonsai macros and the NEON kernel matrix remain pending; no final whole-model performance
or release acceptance is claimed here. The release campaign separately owns Bonsai RAG and 200-public-QA runs.

## Qwen whole-model results

All ten process triplets completed on each host, using 30 distinct measured tasks from two published
datasets. Candidate/control full-logit hashes, greedy tokens and predictions match exactly. All
three arms score 10/15 on JEVBench and 14/15 on SNIPS. These small classification samples are not
a general model-quality certification. The frozen two-percentage-point precision target remains
visible for each metric in the JSON; intervals crossing one remain unresolved.

|Host|Comparison|Metric|Ratio|95% interval|Decision|
|---|---|---|---:|---|---|
|x86|candidate/control|loadNanos|0.998403|[0.987872,1.010582]|unresolved|
|x86|candidate/control|coldRequestNanos|0.870679|[0.840876,0.902648]|faster|
|x86|candidate/control|prefillNanos|0.941344|[0.936143,0.948628]|faster|
|x86|candidate/control|decodeNanos|0.434741|[0.409525,0.468669]|faster|
|x86|candidate/control|requestNanos|0.736578|[0.717617,0.757660]|faster|
|x86|candidate/control|peakRssKiB|0.792326|[0.767443,0.836676]|faster|
|x86|candidate/control|javaAllocatedBytes|0.313934|[0.216378,0.655628]|faster|
|x86|candidate/published|loadNanos|0.992410|[0.979035,1.006479]|unresolved|
|x86|candidate/published|coldRequestNanos|0.872296|[0.842854,0.905792]|faster|
|x86|candidate/published|prefillNanos|0.872070|[0.867769,0.877525]|faster|
|x86|candidate/published|decodeNanos|0.440926|[0.416026,0.473998]|faster|
|x86|candidate/published|requestNanos|0.707044|[0.688684,0.725426]|faster|
|x86|candidate/published|peakRssKiB|0.799786|[0.772724,0.846028]|faster|
|x86|candidate/published|javaAllocatedBytes|0.409223|[0.282297,0.853784]|faster|
|arm|candidate/control|loadNanos|0.994536|[0.977442,1.011364]|unresolved|
|arm|candidate/control|coldRequestNanos|0.798103|[0.774161,0.829332]|faster|
|arm|candidate/control|prefillNanos|0.943313|[0.930403,0.957542]|faster|
|arm|candidate/control|decodeNanos|0.265255|[0.220768,0.338079]|faster|
|arm|candidate/control|requestNanos|0.695882|[0.666294,0.732733]|faster|
|arm|candidate/control|peakRssKiB|0.845535|[0.825517,0.881381]|faster|
|arm|candidate/control|javaAllocatedBytes|0.110693|[0.066441,0.294777]|faster|
|arm|candidate/published|loadNanos|0.995695|[0.976132,1.016987]|unresolved|
|arm|candidate/published|coldRequestNanos|0.897402|[0.871915,0.933429]|faster|
|arm|candidate/published|prefillNanos|1.027957|[1.017914,1.040664]|slower|
|arm|candidate/published|decodeNanos|0.261405|[0.215904,0.337546]|faster|
|arm|candidate/published|requestNanos|0.728551|[0.696190,0.770428]|faster|
|arm|candidate/published|peakRssKiB|0.850963|[0.830505,0.886942]|faster|
|arm|candidate/published|javaAllocatedBytes|0.144344|[0.086672,0.384470]|faster|

ARM candidate/published prefill is a measured regression: 1.027957 [1.017914, 1.040664].
The isolated BF16 change improves the same metric: 0.943313 [0.930403, 0.957542]. Therefore the
BF16 optimization does not explain the regression; exact activation and other shared runtime
changes remain hypotheses until separately measured. Faster total requests do not erase this gate.

## Bonsai continuation protocol

The first measured Bonsai process took roughly five minutes, making the original 30-process phase
longer than its two-hour cap. Before selecting results, the continuation was declared prospectively:
original hosts retain pairs 0–5; immutable v4 hosts run pairs 6–7; immutable v5 hosts run pairs 8–9.
Every triplet stays on one host and retains its original counterbalanced order and tasks. Bootstrap
resampling stays within each host stratum with its observed count. Continuations are bounded to
one hour/$6 each ($12 total), with the campaign 22:32 UTC deadline taking precedence. No trial is
removed based on its performance.

## Revised final artifact qualification

The first complete Bonsai triplet exposes roughly 2.3–2.5 times the published request allocation
in both candidate and corrected control. It does not support release acceptance. Those d92 results
remain historical evidence. The unlaunched v4/v5 continuations are superseded.

The newly frozen runtime is ae337eebccbbae50e750b68babcd58788aeefbf8, jar SHA256
696c2a58843181a2b02c91043208b56c9c6ebdc976e0d3f44dc051cc1f36d555. It adds an explicit callback-close
rejection and moves exact SwiGLU vector work behind an array boundary; the arithmetic stays unchanged.
Eighteen focused executor tests pass, and a child reproduces the old close deadlock and verifies the
fixed rejection. The boundary's full performance and correctness qualification remains pending.

Fresh v6/v7/v8 phases measure every Qwen BF16 and Bonsai triplet on the new artifact: pairs 0–3,
4–6 and 7–9 respectively, each host preserving whole triplets and original counterbalancing. Both
models' real fixtures run before timing. Old and new macro samples are never combined. Per-phase
caps are 90 minutes/$4, combined $12, and the 22:32 UTC campaign deadline overrides these caps.
The separate activation fleet runs exhaustive public SwiGLU and forced-width cold/hot probes on
the same jar hash before its actual-model qualification.
