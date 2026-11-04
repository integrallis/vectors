# Decision latency against thread count on a uniform-core server (the control)

Measured 2026-11-03 on EC2 `m6a.8xlarge`: AMD EPYC 7R13, **16 identical physical cores, 32 logical**,
one socket. Qwen2.5-1.5B-Instruct Q4_K_M, one typed decision, p50 over 15 messages. Raw JMH-style
JSON and the exact worker in `raw/`.

## Why it is here

This is the **negative control** for `2026-11-03-hybrid-cpu-thread-count`. It was run first, to find
out whether thread count explained a 2541 ms decision on an i9-14900HX laptop, and it answered no --
on this machine 32 threads cost 754 ms, a third of the laptop's figure at the same thread count.

That is the finding worth keeping. A uniform-core box **cannot** reproduce the hybrid-CPU problem,
because with interchangeable cores an even split is already an optimal split. Reading this sweep as
though it described the laptop produced one wrong conclusion before the laptop was measured
directly, and the mistake is recorded here rather than quietly dropped.

## Result

| threads | p50 |
|--------:|----:|
| 32 | 754 ms |
| 16 | 496 ms |
| 8 | 752 ms |
| 4 | 1255 ms |
| 2 | 2292 ms |

16 -- the physical core count -- is the optimum here too, and 32 costs 1.52x. So the policy change
(`CpuTopology.computeThreads()`) is right on uniform hardware as well; it is simply worth 1.5x here
against 2.9x on a hybrid part.

The `java-*` and `native-*` arms vary the two pools' thread counts separately. They are **not**
conclusive: the same 32-thread configuration measured 754 ms in one arm and 915 ms in another, about
20% run to run on single runs, which is wider than the differences between those arms. No attribution
between the two pools should be read from them.

## Reproduce

`raw/gguf-thread-worker-0003.sh` is the worker that produced these files, including the host
provisioning. Each arm is one `DecideLatency.java` invocation with
`-Dvectors.gguf.threads` and `-Dmodels.native.kernels.threads` set together.
