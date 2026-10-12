# Exact activation allocation investigation

Actual Qwen3.5 inference on the final combined runtime exposed an allocation
regression that standalone activation microbenchmarks had missed. After 32 warmup
requests, the old exact helper still allocated 190–299 MB per measured request,
compared with roughly 96 MB for the published route.

A separate JFR run (`remote-v1/control.jfr`) identified vector objects and backing
arrays allocated in `ReproducibleActivationExp.exp`, reached through public
`VectorUtil.swiGlu`. The largest stacks were Float-to-Double conversion,
Double-to-Float conversion, and Long-to-Double reinterpretation. JFR sampling
weights describe sampled allocation estimates, not exact byte counts. The
per-request all-thread allocation counter independently confirms the regression.
The diagnostic is separate from decisive uninstrumented timing.

Candidate `100dd4d1` moves the same unfused fdlibm arithmetic into an array-only bulk
SwiGLU method. No Vector object crosses its compiled method boundary. Its useful
body is 551 bytecodes, naturally exceeding the normal hot-inline budget; there is
no padding, compiler flag, private annotation, approximation, or changed contract.
The original helper remains active for causal convolution and the 64-bit fallback.
The Sun notice and bundled notice resource are preserved.

The new public bulk method passed cold/hot, offsets, tails, aliasing, execution
history and concurrent-caller checks locally, plus 10 million random float inputs
against the scalar StrictMath contract. Local timing was diagnostic only. The
supported-host phase uses the actual Qwen3.5-0.8 B model and final-v9 Models artifact;
its 148 cases retain every full-logit snapshot before reusable buffers change.
Candidate and control have identical native libraries and Models JARs.

`activation-boundary-v1` completed the JFR diagnostic, then failed because its
JSON-lines parser tried to parse JFR's pretty-printed JSON. That failed phase is
retained and its host terminated. Immutable v2 fixes only that harness parser.
It runs the same diagnostic, exact 148-case cohort, then 10 counterbalanced process
pairs with 32 warmup requests and 3 measurements per process, capturing startup,
all-thread Java allocation and peak RSS. The final combined artifact also contains
the separately verified BF 16 and executor changes; Q5 final-v3 exercises that
artifact at forced 64/128/256-bit widths, including exhaustive relevant float inputs.

Do not claim all model/resource gates pass from standalone or partial results.

## Completed x86 Qwen 0.8B comparison

All 148 cohort cases and all 30 measured requests retained exact full-logit snapshots.
Boundary/control paired geometric ratios (95% process-pair bootstrap intervals):

| Metric | Ratio | Interval |
| --- | ---: | --- |
| Prefill latency |0.9365|0.9141–0.9589|
| Decode latency |1.0009|0.9977–1.0039|
| Complete model latency |0.9551|0.9386–0.9717|
| Java allocation |0.4578|0.4034–0.5543|
| Peak RSS |0.8502|0.8212–0.8927|
| Fresh model load |1.0049|0.9915–1.0184|

The allocation regime is now stable at 96.23–96.26 MB across all 30 measurements.
The control varies 96.23–299.01 MB; one entire control process pair is already in
the low-allocation regime and remains included. Decode shows no detected change
with a sub-1% interval; startup is not precise enough to resolve 1% and remains
inconclusive. This phase alone does not close ARM or larger-model gates.

Both owned west instances (i-00939a4a2d1fbdde1 and i-06aa47206de734b7f) are confirmed
terminated. `paired-summary.json` and `analyze.py` retain the complete calculation;
raw data is mirrored under `s3://models-qual-077051030817/results/activation-boundary-v2/x86/`.
