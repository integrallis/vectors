# Final runtime startup comparison

This follow-up resolves the startup interval left open by the activation boundary experiment.
It loads the actual Qwen3.5 0.8B model in 300 fresh JVMs: 100 counterbalanced blocks of the
published runtime, the corrected runtime with the prior stable activation, and the final
runtime with the array-only activation boundary. It performs no inference. All arms use the
same model, host, heap arguments, and public backend loader. Each process records model-load
nanoseconds and `/usr/bin/time -v`; load-only RSS is not whole-inference RSS.

Final Models source is `0bc5de31`; final Vectors source is `ae337eeb` (core JAR SHA-256
`696c2a58843181a2b02c91043208b56c9c6ebdc976e0d3f44dc051cc1f36d555`). The stable control
uses the same Models artifacts with Vectors `d92aceb`. The published arm uses the retained
published artifact set and its matching Rust library. Exact artifact hashes are in the
retained `remote/inputs.sha256` and immutable payload manifest.

All 300 processes completed successfully. The final/stable geometric model-load ratio is
**1.00177**, with a process-block bootstrap 95% interval **[0.99759, 1.00597]**: the activation
boundary itself stays within the 1% startup envelope in this experiment. Final/published is
**1.02198 [1.01785, 1.02606]**. That is a startup regression in the accumulated runtime and
is not excused by the activation result. A separate lazy-initialization experiment investigates
an eagerly initialized Q5_1 decoder on models that do not use that format.

Results are in `startup-summary.json`; raw stdout, stderr, classpath, commands, host, and
payload receipts are retained here and at
`s3://models-qual-077051030817/results/activation-startup-v1/x86/`.
Instance `i-0858eac395498a7a2` was verified terminated after the COMPLETE receipt. The phase
had a 15-minute hard cap. The follow-up's raw result remains unchanged regardless of later fixes.
