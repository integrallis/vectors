# BF16 four-row Java allocation correction

This candidate is consolidated into the existing draft BF16 branch for continued work. **It is not qualified for release.** The frozen ten-pair campaign used the older BF16 class; its kernel, NEON and whole-model performance results do not qualify this new class.

JFR identified temporary vector objects escaping through the four-row BF16 projection in actual Bonsai inference. Expanding the packed loads/FMA inside its existing array-only method removes the dominant allocation while preserving every row's even/odd arithmetic and reduction tree. No tuning switch or alternate user-selected route is added.

Two predeclared counterbalanced fresh-process pairs completed on each supported x86/SVE host. Each process executes one cold, one warm and two measured actual Bonsai requests, with eight generated tokens and owned snapshots. All eight requests per architecture match the control's complete nine vocabulary snapshots and tokens exactly. These are diagnostic observations, not a substitute for full qualification.

| Metric, candidate/control | x86 pair 0 / pair 1 | SVE pair 0 / pair 1 |
|---|---:|---:|
| javaAllocatedBytes | 0.392064 / 0.403822 | 0.482120 / 0.490196 |
| prefillNanos | 1.002162 / 1.002605 | 0.997960 / 0.993364 |
| decodeNanos | 0.971724 / 0.969778 | 0.956725 / 0.958600 |
| requestNanos | 0.999684 / 0.999875 | 0.995053 / 0.990987 |
| loadNanos | 0.999700 / 0.998906 | 1.035088 / 1.034674 |
| coldRequestNanos | 1.021964 / 0.999088 | 1.020918 / 0.992671 |
| peakRssKiB | 0.945317 / 0.963858 | 0.982290 / 0.981263 |

Measured allocation falls from 1,242–1,257 MB to 487–507 MB/request on x86 and from 1,460–1,472 MB to 704–722 MB on SVE. **Open gates remain:** x86 prefill observations are 0.22–0.26% slower; ARM load observations are 3.47–3.51% slower; ARM still allocates more than the published runtime. Two process pairs cannot establish the predeclared full no-regression/precision gates. No confidence claim or discarded sample hides these observations.

Correctness and assembly evidence: all twelve local cold/hot shape×width probes pass (64/128/256), including the unchanged 180-second retry after a retained contention timeout. Eleven actual Gradle matrix/dispatch/nested tests pass without skips. The batch allocation probe passes at 9,792 bytes per 1,000 batch-2 calls and 48,000 bytes per 1,000 batch-13 calls. The final actual Gradle JAR passes the pinned Qwen BF16 oracle, fresh-session/reset full snapshots and chunk-boundary batch parity in 21 seconds.

Frozen supported-host diagnostic JAR SHA256: `4320bd5cb517a891517de41704690d52aa47b77ba0497c49a1cd7c07fd67c331`. Final formatted/documented source `9a3e775d` Gradle JAR: `e3e7b0ab79d0630bcad65c4cbe9fd356cf589dd159eba8e6c0aa402db24bd33c`. Its BF16 instructions match the frozen diagnostic exactly; all other runtime classes are identical. Formatting/comments/debug metadata and packaging retain distinct hashes and provenance.

The original single-host diagnostic was prospectively expanded by the parent to one x86 and one ARM host, each capped at 1,200 seconds/$0.50 ($1 combined), absolute 22:30 UTC. East1a ARM capacity failed without creating an instance; East1b succeeded. Both complete raw processes, manifests and failed launch receipt are retained. The redundant contended local full-model process was explicitly stopped with its partial outputs preserved.

Next qualification must use this exact source/artifact: rerun the complete kernel matrix including actual NEON, all supported model fixtures, and the full counterbalanced Qwen/Bonsai macro gates. Investigate ARM's remaining allocation with its actual model caller profile and resolve startup/prefill regressions before release.
