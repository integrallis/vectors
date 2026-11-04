# Thread count on a hybrid CPU: `availableProcessors()` costs 2.88x

Measured 2026-11-03. `thread-count-table.txt` is generated from `raw/` by the script embedded in
this directory's history; every number below traces to a `raw/t-*.log` or `raw/rep*.log`.

## Question

A typed decision on Qwen2.5-1.5B-Instruct Q4_K_M took **2541 ms** on an i9-14900HX laptop while the
same file on an 8 vCPU EPYC server took 754 ms. Thread count was the suspect, but two earlier
guesses at the mechanism were wrong (a missing batched kernel -- it is wired; an unparked Java
executor -- `parkJavaExecutor` is called), so this was measured rather than reasoned about.

## Machine

`Intel(R) Core(TM) i9-14900HX`: **8 performance cores with 2 threads each (cpus 0-15) and 16
single-threaded efficiency cores (cpus 16-31)**, 24 cores and 32 logical processors on one socket.
Three maximum-frequency groups: 16 cpus at 4.1 GHz, 12 at 5.6, and 4 Turbo-Boost-Max-3.0 favoured
cores at 5.8. `governor=powersave`, `driver=intel_pstate` -- the machine's own default, unchanged.
Full topology in `raw/machine.txt`.

## Result

| threads | p50 (3 repeats where shown) | mean |
|--------:|----------------------------:|-----:|
| 32 -- `availableProcessors()`, the shipped default | 2506.0, 2540.6, 2542.7 | **2530 ms** |
| 16 | 1145.4 | 1145 ms |
| 8 -- performance physical cores, the new default | 885.4, 882.5, 863.8 | **877 ms** |
| 4 | 1022.3 | 1022 ms |

**2.88x, within +/-1% across repeats.** The 32-thread arms reproduce the original 2541 ms, so that
first observation was sound. The optimum is exactly the performance-core count: 4 is worse again, so
the lesson is not "fewer threads" but "the number the machine implies".

Accuracy was **11/15 in every arm**. Thread count cannot change an answer -- each output row is its
own dot product folded in a fixed order -- and the measurement confirms it rather than assuming it.

## Why

The native `WorkerPool` partitions work statically by worker index (`model-kernels/src/lib.rs`:
`start_row = rows * worker_index / total_threads`). A barrier then makes a dispatch cost whatever its
**slowest** partition cost, so a single efficiency-core thread holds up all 32 -- on every matrix, in
every layer, for every token.

The reference CPU implementation reaches the same conclusion in its own comments. llama.cpp's
`cpu_count_math_cpus()` (`common/common.cpp`) skips efficiency cores because *"efficiency cores harm
lockstep threading"* and skips SMT siblings because *"hyperthreading isn't useful for linear
algebra"*. On this CPU it picks 8.

## What this does not establish

- The native pool's static partitioning is unchanged in these numbers. This measures the **thread
  count** alone, against published `0.1.76`. Chunk stealing may add more on top, or little; on a
  homogeneous EPYC it changed nothing at 32 threads, which is why no uniform-core box could have
  found this.
- `governor=powersave` throughout. A `performance` run was not taken, so these are the machine's
  default-state numbers and a floor rather than a ceiling.
- Three pools still size themselves independently on the `rust-ffm` path (the vectors GGUF executor,
  the native `WorkerPool`, and models' SwiGLU pool). Only the first is fixed here.

## Reproduce

```
jbang -Dvectors.gguf.threads=32 -Dmodels.native.kernels.threads=32 DecideLatency.java
jbang -Dvectors.gguf.threads=8  -Dmodels.native.kernels.threads=8  DecideLatency.java
```

`DecideLatency.java` is `models-bench/jbang/DecideLatency.java` in the models repository. The
detected policy for any machine is `VectorUtil.computeThreads()`; `raw/cpu-topology-probe.txt` is
its output on this one.
