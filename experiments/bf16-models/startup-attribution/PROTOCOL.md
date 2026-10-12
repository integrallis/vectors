# BF16 startup attribution

The frozen final ten-pair x86 macro result detects a candidate/control startup regression: 1.015080 [1.003587, 1.026042]. That failed result remains unchanged.

A separate prospective phase uses 100 counterbalanced fresh-process Qwen loads on one x86 host, alternating control/candidate order. Both arms retain Models 0bc and the same immutable artifacts from bf16-models-v9; only the BF16 class differs. The macro harness initialization, task parsing, configuration and measured load boundary are retained; startup mode returns immediately after the existing load measurement. This phase does not perform inference or replace whole-model latency evidence.

Record two separate class-load/initialization traces, then retain all 100 pairs. Analyze geometric process-paired ratios with 20,000 percentile bootstrap resamples, seed 716309, 95% interval and the existing two-percentage-point precision target. Intervals crossing one remain unresolved; no trial exclusion or threshold adjustment is allowed.

Root approved at most 15 minutes/$0.30, with 22:30 UTC absolute cutoff. The actual launched worker is more tightly capped to 780 seconds and 21:59:50 UTC, using the existing on-demand pricing plus conservative overhead allowance. It terminates automatically and has a separate ownership-checked deadline monitor. Payload SHA 275c14daa0a60ec447377a71818aea2b82bfa3da74550cce8ae3457ab277f05b was frozen before launch. This is an attribution phase; new findings must not silently reclassify the failed preceding macro result.
