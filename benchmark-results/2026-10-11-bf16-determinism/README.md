# BF16 cold/compiled regression

The public BF16 matrix API changed results as `reduceLanes(ADD)` moved from interpreted to
compiled execution. The new fresh-JVM test failed in four of the original nine configurations
before the fix (`red.xml`). An explicit fixed-order reduction eliminates the drift. The final
suite covers twelve configurations (`final-BFloat16MatrixColdHotTest.xml`) and an allocation
regression (`final-BFloat16MatrixAllocationTest.xml`). Each correctness probe compares raw
float bits, both tokens in offset batched input/output, and 256 cold-to-warm iterations.

Shapes cover Bonsai gates (48 × 5120), grouped rows plus scalar tails (5 × 37), tall matrices
(8193 × 37), and the parallel threshold (512 × 2048). Width ceilings are 64, 128, and 256 bits;
actual width is capped by host support. The local host is Intel macOS, Temurin 25.0.3, four
active processors. Synthetic cases isolate arithmetic stability, not model quality.

## Measured implementations

The first fix reused `PanamaVectorUtilSupport.reduceLanesFixedTree`. Correctness passed, but
ABBA JMH measurements (`paired-jmh/`) found a small-batch latency ratio of 1.2124. Profiling
showed about 39,941 allocated bytes per 13-token gate call versus about 5 in the baseline.
The JVM declined to inline the nested reduction helper; each row/token accumulator escaped.
That implementation was rejected rather than accepted on numerical correctness alone.

- Reusing the four-row matvec path recovered speed but still allocated about 29,955 bytes/call.
  The batch-routing change is not retained.
- Making the reduction width constant and splitting the helper into smaller methods did not
  eliminate boxing. Neither shared-helper experiment is retained.
- Keeping the fixed tree directly in `BFloat16Matrix.finishRow` restores the original batch
  routing and avoids the nested call. The focused profile measures about 4 bytes/call and
  549 microseconds, versus 592 microseconds for the baseline in that diagnostic pair.
  These profiler runs explain the mechanism; `final-paired-jmh/` is the uninstrumented-inlining,
  GC-profiled ABBA performance comparison.

The final allocation probe passes with both the original baseline and retained implementation,
and fails with the first fixed-tree helper implementation: about 6.15 MB over 1,000 two-token
calls. This specifically guards batched projections. A preliminary single-vector probe found
about 2.3 KB/call in both the original and fixed implementations; it did not establish a new
single-vector allocation regression.

Patches, jar hashes, commands and inlining excerpts retain the rejected experiments. No JVM
inlining override or new batch dispatch rule is required by the retained code.

## Real model and release scope

With only the corrected Vectors core classes prepended to the existing Models test classpath,
both complete native Bonsai fixture tests pass: the publisher continuation/fresh sessions and
zero-error full-vocabulary sequential/batched logits. The separate 16-token state probe also
matches recurrent, convolution, key and value state arrays. This is local source validation,
not a released-artifact qualification. The final Models checks must consume released Vectors
on Linux x86 and ARM before their CI failure is considered resolved.

The benchmark build also exposed stale `jmhRuntimeClasspath` and benchmark `runtimeClasspath`
dependency locks. Regenerating them restores the JMH artifact build; CI now builds it on every
PR. Full build, portability CI, and released dependency-chain validation remain required.
