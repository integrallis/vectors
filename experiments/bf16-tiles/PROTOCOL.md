# BF16 fixed-order tiling and bounded dispatch

The previous six-pair row-layout matrix rejected every candidate: direct reduction is faster
on x86 but slows ARM tall matvec and prompt batching. Inline four-row reduction improves ARM
ordinary matvec, while two-row vocabulary streaming is slower on both architectures. All
failures remain evidence; none is an accepted release path.

The next hypotheses reuse the existing bounded executor for large, shareable BF16 matvec,
and decode each BF16 weight vector once for two or four prompt activations. Each activation
retains the original sequence of even then odd FMA operations and the fixed reduction tree.
No extra threads, architecture dispatch, new pool, or numeric tolerance are introduced.
Confined segments and matrices below the existing work threshold execute on the calling thread.
The serial four-query arm isolates batching from dispatch.

Before performance, all candidate artifacts must pass the twelve cold/hot shape/width probes
and the existing batch allocation probe. Performance retains all six previous workload/shape
combinations, six counterbalanced process pairs, five one-second warmups and five one-second
measurements, GC profiling, JDK, four processors, and paired bootstrap analysis (20,000 resamples,
seed 716309). Compare baseline, tile2parallel, tile4parallel, and tile4serial. A detected
slowdown is a failure; an interval crossing one is unresolved, never automatically a pass.
Retain every valid sample, exact commands, source and artifact hashes. A failure in correctness
or allocation rejects that candidate before timing, while other candidates may continue; record
which candidates were rejected and do not label an incomplete matrix complete.

This is a kernel experiment. A production selection still requires actual model fixtures,
whole-model comparisons, complete tests, and dependency-consumer checks. Two dedicated hosts,
x86 and ARM, have a one-hour limit and combined $2 ceiling under the existing fleet launcher.

Local construction checks rejected v1 because the combined tile/remainder method allocated
roughly 2.95 MB per batch call, and v2 because the separate remainder reduction still boxed
about 3 KB per call. V3 puts the fixed reduction directly in both tile and remainder methods.
A superseded v2 four-query probe was stopped after its tile source was revised; it provides no
performance evidence. The final candidate's checks run separately on immutable packaged bytes.
