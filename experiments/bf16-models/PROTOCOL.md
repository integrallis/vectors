# Actual-artifact BF16 model completion

Use the pinned Qwen 2.5 0.5B BF16 Safetensors weights and Bonsai 2 PTQ GGUF. Verify every
artifact checksum before load. Keep three immutable arms on each of one x86 and one ARM host:
published Models 4337ee45 with published Vectors; corrected native Models 12796ebf with fixed
BF16 reductions and the selected exact activation code; and those same corrected Models and
activation bytes with the two-query BF16 tile and bounded matvec dispatch. No runtime route
flag or instrumentation changes the measured implementation. The control isolates the BF16
layout/dispatch change; the published arm exposes regressions against the shipped stack.

The final artifact also repeats the complete six-shape BF16 kernel matrix using twelve
prospectively declared counterbalanced process pairs. The earlier six-pair two-query candidate
left the ARM small matvec interval crossing one; that result remains inconclusive. The final
artifact uses the existing `HAS_SVE` capability to inline the four ordinary-row reductions
on SVE and retain direct helper reduction elsewhere. Previous paired measurements found the
SVE inline ratio 0.9657 [0.9606, 0.9699], while direct helper was inconclusive there; on x86
direct helper measured 0.7565 [0.7475, 0.7662], while inline was inconclusive. The arithmetic
tree is identical in both layouts, and there is no new route property. The final matrix
validates this prospective hardware policy, and also includes the nested-executor fix. Neither previous source is represented as this
artifact. All twelve cold/hot probes, allocation and nested dispatch run before kernel timing.

Run real-model fixture gates before the kernel timing matrix so a model failure is visible
without waiting for the full microbenchmark. A candidate or corrected-control fixture failure
blocks every performance measurement; complete the independent remaining fixtures, preserve
the failure reports, and terminate the host. A published-arm failure remains historical evidence.

Before timing, record all existing fixture gates for all arms, retaining published failures.
The candidate and corrected control must pass: pinned Qwen Transformers logits at the existing
1e-3 tolerance, pinned ChatML token IDs, vocabulary size and winning token; pinned Prism Bonsai
eight-token continuation and input token IDs; observable batched-prefill routing; implicit-session
reset parity; repeated fresh
sessions; exact full-vocabulary sequential versus 16-token prefill across a chunk boundary.
The Bonsai oracle uses the existing native test's 64-token context and its batch-parity case
uses 96 tokens, reproducing both native integration cases. Timing separately declares 2048.
The native Bonsai projection route must actually be enabled. Failure is recorded and blocks
acceptance, while independent remaining model coverage continues.

Timing uses the same protocol as the current actual-artifact GDN comparison: ten counterbalanced
independent process pairs for every model, four initial requests (the first separately retained
as cold), then three measured requests per process. Thirty measured requests alternate the
published JEVBench/SNIPS datasets (fifteen each) using the existing frozen workload and eight
greedy output tokens. Run library default worker/polling/dispatch settings, with a declared
2048-token context memory budget and identical JVM heap flags in all arms. Record load time,
cold request, prefill, decode, total request, maximum RSS, all-thread Java allocations, every
full-logit digest, greedy token, predicted label, and expected label. No observations are discarded.

Analyze geometric means of process-paired ratios with paired percentile bootstrap, 20,000
resamples, seed 716309; inspect each model and metric separately. A detected slowdown or
correctness failure blocks acceptance; confidence intervals crossing one remain unresolved.
The desired interval width is two percentage points. Do not widen gates to rescue a candidate.
Root owns the additional final Bonsai RAG and public-QA qualification and dependency releases.

Two owned hosts have a two-hour deadline and combined $3.50 cap. Use the existing quota/price
checks, immutable payload/commit receipts, periodic diagnostic upload, and confirmed termination.

Oracle values are copied without changing tolerances from Models
`Qwen2HuggingFaceBackendIntegrationTest` and `Bonsai2NativeModelFixtureSlowTest`.
The frozen payload contains the exact harness and pinned artifact hashes.

The fixture/probe output records the ordinary-row reduction branch derived from the immutable
SVE capability used by the production dispatch. Non-SVE ARM performance is not established by
this pair of hosts; forced widths on the SVE host establish correctness only.

Final BF16 qualification uses the stable exact activation implementation from `a5511d6`.
Rejected polynomial candidates are retained separately and are not present in this payload.
Any later activation replacement requires fresh whole-model evidence on that changed artifact.

The closed-executor follow-up repeats every model fixture, cold/hot, allocation and nested
correctness probe on its new artifact. Its BF16 class must be byte-identical to the v1 artifact
before reusing v1 kernel timing; full model timing is always repeated. The executor now rejects
nested work on a closed pool, with volatile visibility of closure. The v1 macro harness retained
backend-owned logit arrays and therefore cannot prove per-step hashes or first-token label
scores. Fixture snapshots already cloned before every forward call, so those fixture results
remain valid. Follow-up macro snapshots clone immediately; measured snapshot-copy wall time and
main-thread allocations are retained separately and subtracted from observed request/all-thread
totals. Prefill/decode timers exclude capture, and all arms run the same corrected harness.
