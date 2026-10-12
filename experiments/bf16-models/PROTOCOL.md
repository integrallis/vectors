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
artifact inlines the four ordinary-row reductions, motivated by the previously measured ARM
benefit, and includes the nested-executor fix. Neither previous source is represented as this
artifact. All twelve cold/hot probes, allocation and nested dispatch run before kernel timing.

Run real-model fixture gates before the kernel timing matrix so a model failure is visible
without waiting for the full microbenchmark. A candidate or corrected-control fixture failure
blocks every performance measurement; complete the independent remaining fixtures, preserve
the failure reports, and terminate the host. A published-arm failure remains historical evidence.

Before timing, record all existing fixture gates for all arms, retaining published failures.
The candidate and corrected control must pass: pinned Qwen Transformers logits at the existing
1e-3 tolerance and winning token; pinned Prism Bonsai eight-token continuation; repeated fresh
sessions; exact full-vocabulary sequential versus 16-token prefill across a chunk boundary.
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
