# Deterministic activation arithmetic screening

The source StrictMath activation correction passes cold/hot correctness, but whole-model
confirmation shows slower execution than the published arithmetic. That comparison also
includes a projection correction, so it does not isolate the activation cost.

This experiment compares the published Vector API exponential, the current scalar StrictMath
lane implementation, and an unfused double-SIMD translation of fdlibm narrowed to float inputs
and outputs. The original permissive source notice is retained. No approximation tolerance is
added: the candidate must reproduce the float cast of StrictMath.exp exactly.

Correctness: ten million deterministic random float bit patterns at each supported capped width,
then all 580,911,104 signed float bit patterns with magnitude from 2^-28 through 104. This covers
the nontrivial finite float-output exponential range. The random probe also includes special
values, tiny inputs and overflow/underflow cases. Record actual species width; a capped width
above the host's preferred width does not claim wider native hardware coverage.

Timing: JMH, three fresh forks, five warmup and ten measurement iterations of one second each,
all three implementations and lengths 32, 4096 and 16384. Inputs are fixed public-seed uniform
values in [-10,10]. Record GC allocation as well as latency. These synthetic component timings
select a candidate for real-model testing, not a production winner. Real model correctness and
paired latency measurements on both public datasets remain required.

Run dedicated x86 and ARM hosts, pin JDK and payload hashes, limit each host to 30 minutes and
the pair to $1.25, upload diagnostics immediately and confirm owned-instance termination. A
failed or incomplete check remains failed or incomplete.

## Follow-up after the first dedicated-host run

The first candidate preserves the exhaustive arithmetic contract on both hosts but is slower
on x86. GC profiling records 589,824 allocated bytes per 4,096-input call on x86 and 131,072
on ARM. It is rejected as a production replacement.

The next candidate uses integer conversion for bounded exponents, uniform range reduction
verified over the same complete float range, and one double-vector calculation per call. The
float input vector uses half the double vector's bit width so both hold the same number of
values. The smallest supported width uses scalar StrictMath when a one-lane float vector would
be required. The baseline benchmark retains its original full float width.

Local compiler logs for the intermediate two-group implementation report
`NodeCountInliningCutoff` at the second double-vector call and retain 65,536 bytes per call.
The single-group implementation inlines the calculation and reduces that measurement to fixed
profiler overhead (0.24 bytes per call); all 580,911,104 checked input patterns still match.
These are local diagnostics, not a supported-host performance verdict. The second immutable
payload repeats the same dedicated-host correctness matrix and full JMH protocol.

## Production API candidate

The selected screening implementation is integrated into `swiGlu` and
`causalDepthwiseConv1dSilu` through a private `ReproducibleActivationExp` helper.
The original full-width loop bound is retained so a changed vector width cannot
move values between fused vector convolution and the existing scalar tail.
The small sigmoid/softplus operation retains the stable StrictMath implementation.
The fdlibm permission notice is preserved both in source and `META-INF/NOTICE`.

On the dedicated x86 and ARM hosts the second immutable screening payload passes
all 580,911,104 nontrivial finite input patterns. At 4,096 inputs its component
latency is 13,713.57 ns on x86 and 16,811.04 ns on ARM, versus 49,968.64 and
64,191.01 ns for scalar StrictMath lanes. The published, compilation-sensitive
Vector API exponential remains faster (2,193.40 and 4,520.21 ns). These are
component results, not a claim that the full model meets its performance gates.
Raw host results are retained in the Models proof campaign's
`remote-activation-exp-v2/{x86,arm}/jmh.json` and `exhaustive.log`.

`ActivationPublicBenchmark` also exercises the actual public methods, including
causal history updates. Local short-run comparisons and their exact commands are
retained as diagnostics; the supported-host whole-model comparison is the
acceptance test. Numerical assertions and performance gates are unchanged.
