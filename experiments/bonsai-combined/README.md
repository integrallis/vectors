# Bonsai correction: combined runtime

The candidate includes both reproducible activations and fixed BF16 reductions.
Neither component alone covers the observed failures. The full activation-only
candidate passes the original two native JUnit cases, but fails the broader
fresh-session check on both Linux architectures: full logits change while selected
tokens still match. BF16-only fails the original batch-parity case. The combined
runtime passes both sets of numerical checks.

The combined correction before the serial-allocation fix is Vectors `6893a587`, SHA-256
`e3e7b0ab79d0630bcad65c4cbe9fd356cf589dd159eba8e6c0aa402db24bd33c`.
[Original unchanged native fixtures](https://github.com/integrallis/models/actions/runs/38319081991)
passed on both architectures. The [whole-model run](https://github.com/integrallis/models/actions/runs/38321278750)
also requires full-vocabulary equality across fresh sessions before timing.
Its completed Qwen measurements improve latency, but precision bounds remain
unresolved; Bonsai measurements are still running. A separate fixed 400-pair
startup study per architecture passes the declared one-percent load allowance in
[38323913262](https://github.com/integrallis/models/actions/runs/38323913262).
ARM's measured increase is 0.39%, within that allowance, rather than zero.

The current candidate additionally avoids allocating a captured row callback for
serial BF16 batches, without changing their arithmetic or threading threshold.
Its core JAR is byte-identical to Vectors `91dd64ae`, SHA-256
`13f0bd516ede8e4306541c72ef368e94c9bddddd80a99631ff5aa65992fdbba8`.
Only `BFloat16Matrix.class` differs from the prior combined core. The expanded
Bonsai suite passes on Linux x86 and ARM with Temurin 25.0.3 and 25.0.4.1 in
[38324642561](https://github.com/integrallis/models/actions/runs/38324642561):
three tests per configuration, zero failures, errors or skips.

The allocation ablation is [38324610747](https://github.com/integrallis/vectors/actions/runs/38324610747).
ARM serial batching removes about 48 bytes per call, but the parallel timing
interval crosses parity; that diagnostic does not pass all its frozen gates.
On x86 the serial timing interval also crosses parity and includes a 4.39%
slowdown; no allocation reduction was observed there. This is unresolved,
not evidence that the change has zero cost on every host.
Final qualification must measure the current artifact against published Vectors.
The earlier artifact's whole-model results are retained with their original scope.
Nothing has been published.

The activation-only failure is retained in [run 38321902032](https://github.com/integrallis/models/actions/runs/38321902032).
Its pinned JDK was Temurin 25.0.3; the earlier original-fixture workflow used floating
Temurin 25 (currently 25.0.4.1). Both patch versions require regression coverage.

The scalar portability failure was in the newly added test oracle: SIMD-specific
rounding was expected even when scalar mode was selected. The corrected test uses
the actual provider's existing exact contract and includes explicit scalar cases.
Production arithmetic and original Bonsai assertions are unchanged by this test fix.
