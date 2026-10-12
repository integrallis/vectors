# Bonsai correction: combined runtime

The candidate includes both reproducible activations and fixed BF16 reductions.
Neither component alone covers the observed failures. The full activation-only
candidate passes the original two native JUnit cases, but fails the broader
fresh-session check on both Linux architectures: full logits change while selected
tokens still match. BF16-only fails the original batch-parity case. The combined
runtime passes both sets of numerical checks.

The current core JAR is byte-identical to Vectors `6893a587`, SHA-256
`e3e7b0ab79d0630bcad65c4cbe9fd356cf589dd159eba8e6c0aa402db24bd33c`.
[Original unchanged native fixtures](https://github.com/integrallis/models/actions/runs/38319081991)
passed on both architectures. The [whole-model run](https://github.com/integrallis/models/actions/runs/38321278750)
also requires full-vocabulary equality across fresh sessions before timing.
Its completed Qwen measurements improve latency, but precision and startup bounds
remain unresolved; Bonsai measurements are still running. These are not passing
release gates yet.

The activation-only failure is retained in [run 38321902032](https://github.com/integrallis/models/actions/runs/38321902032).
Its pinned JDK was Temurin 25.0.3; the earlier original-fixture workflow used floating
Temurin 25 (currently 25.0.4.1). Both patch versions require regression coverage.

The scalar portability failure was in the newly added test oracle: SIMD-specific
rounding was expected even when scalar mode was selected. The corrected test uses
the actual provider's existing exact contract and includes explicit scalar cases.
Production arithmetic and original Bonsai assertions are unchanged by this test fix.
