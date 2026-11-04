# GGUF stage-plan publication: Phaser barrier vs generation/countdown

Measured 2026-11-03, same machine and protocol as
`2026-11-03-gguf-dispatch-barrier`. Raw artifacts in `raw/`; `stage-table.txt` is generated from
`raw/stage.json` by `scripts/summarize-dispatch-ab.py`.

## Why this run exists

The companion run found the barrier mechanism 2.0-3.6x cheaper in isolation but measured **no
resolvable difference** on a 2048x2048 row-parallel projection. That null has a cause:
`GgufParallelSupport.forEachRow` refuses to go parallel below 1,048,576 weight elements, so a row
dispatch always carries hundreds of microseconds of arithmetic and a 2 us handshake disappears into
it.

`GgufStagePlan.execute()` has **no size gate at all** and pays a barrier between every pair of
stages. `QuantizedBatchedLayerPlan` in models uses it once per layer. That is the path where the
handshake is a real fraction of the dispatch, so it is the path that decides whether the mechanism
is worth anything.

## Machine

AMD EPYC 7R13 (Milan), 8 vCPU = 4 physical cores x 2 threads, EC2 `m6a.2xlarge`, Temurin JDK 25.
`raw/machine.txt` has the full `lscpu`. JMH 1.37, 3 forks x (5 x 1 s warmup + 5 x 2 s measurement).

Two stages of 152 and 1024 work items, the second depending on the first — the existing
`GgufStagePlanBenchmark` shape, parameterized over implementation and thread count rather than
fixed at a persistent pool of 8.

## Result

| threads | `stagedDispatch` persistent (us) | countdown (us) | ratio |
|--------:|---------------------------------:|---------------:|------:|
| 1 | 1.441 | 1.340 | no difference resolved |
| 2 | 1.718 | 1.657 | 1.04x |
| 4 | 2.476 | 1.579 | 1.57x |
| 8 | 4.453 | 1.721 | **2.59x** |

`separateDispatches` (the same work as two independent `forEach` calls) agrees and is slightly
larger: 1.06x at one thread rising to **2.76x** at eight.

## The mechanism, confirmed

The ratio is not the interesting part. This is:

| benchmark | implementation | 1 thread | 8 threads | cost change |
|---|---|--:|--:|--:|
| `stagedDispatch` | Phaser | 1.441 | 4.453 | **3.09x worse** |
| `stagedDispatch` | countdown | 1.340 | 1.721 | 1.28x worse |
| `separateDispatches` | Phaser | 1.560 | 5.583 | **3.58x worse** |
| `separateDispatches` | countdown | 1.475 | 2.024 | 1.37x worse |

Publishing a job to eight threads costs the Phaser implementation three and a half times what
publishing to one thread costs it. That is the predicted failure: arrival count and release signal
share one word, so every added party both CASes the line and spin-reads it, and the handshake takes
back part of what the extra threads earn. Splitting the two roles onto separate lines leaves the
cost nearly flat in thread count, which is what a publication mechanism should do.

Both implementations were predicted to behave this way before the run, and both did.

## What this does and does not establish

Established: on an ungated dispatch path, at the thread count an ordinary 8-vCPU deployment uses,
the new mechanism is 2.6x cheaper, and the old one's cost grows with parallelism while the new one's
does not.

**Not established: any end-to-end token-rate figure.** These are microbenchmarks of the handshake.
The saving is about 2.7 us per dispatch at eight threads; whether that is visible in tokens per
second depends on how many ungated dispatches a real model runs per token, which this run does not
measure and about which no claim is made here. The decode and prefill A/B is the next step, and
until it runs the row-path null above stands as the only statement about the matmul path.

## Reproduce

```
./gradlew :vectors-bench:jmhJar
java -jar vectors-bench/build/libs/vectors-bench-*-jmh.jar \
  GgufStagePlanBenchmark -rf json -rff stage.json
python3 scripts/summarize-dispatch-ab.py stage.json
```

`raw/gguf-barrier-worker-0002.sh` is the exact worker that produced `raw/stage.json`.
