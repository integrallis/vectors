# BF16-only dependency candidate

This candidate starts at published v0.1.29. Runtime changes are confined to BF16 arithmetic,
weight reuse, existing row dispatch, and the persistent executor safety required by that dispatch.
PanamaVectorUtilSupport changes only expose its existing fixed-tree shuffle constants. Activation
code, scalar arithmetic, platform detection and all other kernels remain published code.

The initial minimal fixed-tree checkpoint 85b344c7 is a correctness diagnostic, not a release
candidate: the retained six-pair experiment found regressions on ARM and small x86 prompt batches.
The subsequent BF16 implementation incorporates the previously measured two-query weight reuse,
large shared-segment row dispatch, capability-based reduction layout, and array-only four-row
unpack/FMA body. No previous timing result qualifies this newly combined artifact.

The supported GitHub runner matrix records CPU identity, JDK, source and actual core JAR hashes.
Run unchanged cold/hot raw-bit probes at 64/128/256 bits, allocation, confined/shared memory,
concurrent and nested dispatch contracts before timing. Compare the actual candidate Gradle JAR
against the actual v0.1.29 Gradle JAR, using one unchanged existing BF16 JMH benchmark compiled
against the baseline. It is not a production classpath overlay.

Freeze twelve independent process pairs prospectively, alternating baseline/candidate and
candidate/baseline. Each process uses four visible processors, five one-second warmups, five
one-second measurements, one fork, a 3 GiB heap and the GC profiler. Retain all six combinations
of matvec/prompt batch and 48x5120, 4864x896, 151936x896 shapes. Retain every valid sample;
failed or missing samples leave qualification incomplete. Analyze geometric means of paired
ratios with 20,000 paired bootstrap resamples and seed 716309. An interval wholly below one is
faster, wholly above one is slower, and crossing one is unresolved; unresolved is not a pass.
The 95% interval width must also be no greater than 0.02. Report absolute allocation for both
arms; the unchanged focused allocation contract remains a prerequisite.

A successful workflow proves only these kernel cases on the recorded hardware. Hosted runners
may lack SVE and do not stand in for unmeasured SVE systems. Actual unchanged Bonsai and Qwen BF16
fixtures, whole-model resource/performance comparisons, full build and normal dependency-pinned
consumer CI remain required before publication. No release/tag/publish action is in this workflow.

Historical protocol audit: the earlier `experiments/bf16-tiles/run-host.py` six-pair v2 script
rotated a two-arm list and then reversed odd pairs. Those two operations canceled, so its actual
order was baseline then candidate in every pair, despite its description as counterbalanced.
That historical result does not qualify this candidate. The present script explicitly alternates
the two orders and records every executed command. The older four-arm masked-reduction experiment
is a separate data set and is not affected by this two-arm ordering defect.
