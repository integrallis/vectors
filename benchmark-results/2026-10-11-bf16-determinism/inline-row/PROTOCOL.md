# BF16 row-layout investigation

The six-pair masked reduction experiment rejects every candidate as a universal replacement:
each has a detected slowdown on at least one host/shape. All original results and failures stay
retained. In particular, the masked candidate slows two x86 matvec shapes despite passing the
real Bonsai correctness fixtures. No performance tolerance is widened.

The same GC profiles measure approximately 48 allocated bytes per row for the existing four-row
matvec on both hosts (about 2,304 bytes at 48 rows and 233,472 bytes at 4,864 rows). The short local
diagnostic removes that allocation by placing the fixed reduction inside the accumulation loop,
but does not prove a speedup. A two-row small-matrix diagnostic is slower locally and is not
selected. The separate direct-reduction candidate reuses the existing reduction helper directly
at the accumulation call site, with only the reduced float crossing the scalar-tail boundary.
It preserves the local cold/hot and batch-allocation checks but still allocates in matvec.

This next experiment compares baseline, original rotate correction, direct reduction, and
inline reduction with four-row small matrices/two-row tall matrices. The last candidate retains
the existing row-count boundary; its tall-row change is a hypothesis about activation reuse and
memory streaming, not a measured production routing recommendation.

Use the same six counterbalanced process pairs, all six benchmark/shape combinations, five
one-second warmup and five one-second measured iterations, GC profiling and JDK as the masked
experiment. Each candidate must first pass all twelve cold/hot configurations and the batch
allocation probe. No valid trial is discarded. Compute paired confidence intervals with
process pairs as the independent unit; detected slowdowns remain failures, and intervals
crossing one remain unresolved. Retain all source/JAR hashes and commands.

Dedicated x86 and ARM hosts have a one-hour deadline and a combined $2 ceiling. Verify live
quota/pricing, upload diagnostics during execution and confirm owned-instance termination.
Passing this synthetic matrix alone does not qualify a release: real model fixtures, whole-model
performance, full tests and released dependency-chain checks still apply.
