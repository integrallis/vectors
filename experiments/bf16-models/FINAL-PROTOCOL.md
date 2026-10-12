# Final combined-runtime model proof

Frozen runtime: Vectors ae337eeb, core jar SHA256
696c2a58843181a2b02c91043208b56c9c6ebdc976e0d3f44dc051cc1f36d555,
and actual Models 0bc5de31 release jars copied from the immutable final-v11 payload.
Published comparator remains Models 4337ee45 and published Vectors.
Corrected control differs from candidate only in the original deterministic BF16 kernel class.

The previous v6 runtime used Models 12796 and is superseded before qualification. Its collected
outputs remain separate; v7/v8 never launched. New v9/v10/v11 phases contain every Qwen BF16 and
Bonsai process triplet on the final Models/Vectors combination: pairs0–3,4–6,7–9 respectively.
No earlier-runtime sample enters final analysis. All triplets remain on the same host, including
published, corrected control and candidate. Counterbalanced order, tasks, repetitions and gates
are unchanged. Bootstrap resampling is stratified by host.

Each phase is capped at90minutes and$4 across both hosts; all three phases total at most$12.
The22:32UTC deadline overrides host caps. One critical ARM host uses West Spot capacity with its
instance/region/AMI recorded. Other hosts use reserved East capacity. Spot price is bounded using
the on-demand price in the budget preflight. Any interrupted/incomplete triplet remains incomplete.
Real-model fixtures and kernel correctness/allocation/lifecycle probes precede model timing.
The activation fleet verifies exhaustive public SwiGLU and forced-width cold/hot probes on the same
Vectors jar. Kernel timing reuse requires byte-identical BF16 class content.
