# GGUF row-executor publication: Phaser barrier vs generation/countdown

Measured 2026-11-03. Raw artifacts in `raw/`; `dispatch-table.txt` is generated from
`raw/dispatch.json` by `scripts/summarize-dispatch-ab.py` and is not typed by hand.

## Hypothesis

`GgufPersistentRowExecutor` publishes a job and observes completion with a `Phaser`. A Phaser keeps
the arrival count and the release signal in one word: every party CASes that word in `arrive()`
while every waiting party spin-reads it in `getPhase()`, so each of the N-1 pollers invalidates the
cache line the arrivers need to own. A `forEach` pays that twice, once to publish and once to
complete.

If that is the dominant cost, then splitting the two roles onto separate cache lines should reduce
per-dispatch cost, and reduce it *more* at higher thread counts. `GgufCountdownRowExecutor` does
that: workers poll a generation counter the publisher writes exactly once per dispatch and no CAS
ever touches, and completion is a countdown each participant decrements once with only the caller
polling it. It also removes the second barrier, because a worker that finishes early can loop back
and wait for the next generation instead of meeting its peers.

Chunking, the poll-before-park budget and caller participation are identical between the two, so the
comparison isolates the publication mechanism.

## Decision rule, fixed before the run

- The handshake benchmark (`emptyDispatch`) decides whether the mechanism is cheaper at all.
- `projectionDispatch` decides whether it matters beside real arithmetic. **If the projection shows
  no resolvable difference, that is the reported result for the row-matmul path** — not a reason to
  go looking for a kinder workload.
- Overlapping JMH error bars are reported as "no difference resolved", never as a win for the lower
  number.

## Machine

AMD EPYC 7R13 (Milan), 8 vCPU = 4 physical cores x 2 threads, 2.6 GHz, EC2 `m6a.2xlarge`,
Ubuntu 24.04, Temurin JDK 25. Full `lscpu` in `raw/machine.txt`. Chosen as an ordinary cloud
deployment shape rather than a large box, and the same CPU family as the existing grouped-decision
measurement so the numbers sit beside it.

JMH 1.37, 3 forks x (5 x 1 s warmup + 5 x 2 s measurement) per configuration.

## Result

The handshake is substantially cheaper, and the gap widens with thread count exactly as predicted:

| threads | persistent (us) | countdown (us) | ratio |
|--------:|----------------:|---------------:|------:|
| 1 | 0.079 | 0.033 | 2.36x |
| 2 | 0.425 | 0.313 | 1.36x |
| 4 | 1.160 | 0.512 | 2.27x |
| 6 | 1.831 | 0.593 | 3.09x |
| 8 | 2.666 | 0.738 | 3.61x |
| 12 | 9.043 | 6.637 | 1.36x |

`smallDispatch` (128 rows) agrees: 2.02x at one thread rising to 3.22x at eight.

At 12 parties on 8 vCPUs both implementations degrade by roughly an order of magnitude. Parking and
yielding dominate an oversubscribed pool, and the mechanism is no longer what is being measured;
countdown keeps a 1.36x edge there but the honest reading is that neither is good in that state.

## The null, which is the more important half

`projectionDispatch` — a 2048x2048 row-parallel projection, the shape a decoder runs several times
per layer — shows **no resolvable difference at any thread count**. Error bars overlap at 1, 2, 4,
6, 8 and 12 threads.

That is not a surprise once the production gate is read: `GgufParallelSupport.forEachRow` refuses to
go parallel below `DEFAULT_MIN_ELEMENTS` = 1,048,576 weight elements, so every dispatch that reaches
the pool through the row path already carries hundreds of microseconds of arithmetic. A 2 us saving
on a 450 us dispatch is not measurable and would not be worth having if it were.

**For the row-matmul path, this change is a null.** It should not be sold as a decode speedup.

## Where it should still matter, and what has not been measured

Two production paths are not gated at a mebibyte:

- `GgufStagePlan.execute()` has no size gate at all and pays a barrier between every pair of
  stages. `QuantizedBatchedLayerPlan` in models uses it once per layer.
- `GgufParallelSupport.forEachPackedInt4Unit` gates at 65,536 elements, 16x smaller than the row
  path, which puts dispatches in the tens of microseconds where a 2 us handshake is a real
  fraction.

Those are where the measured handshake improvement could convert into throughput. **Neither has
been measured here.** The stage path is measured separately in
`2026-11-03-gguf-stage-barrier`; the packed-int4 path remains unmeasured and no claim is made
about it.

## Reproduce

```
./gradlew :vectors-bench:jmhJar
java -jar vectors-bench/build/libs/vectors-bench-*-jmh.jar \
  GgufRowExecutorDispatchBenchmark -rf json -rff dispatch.json
python3 scripts/summarize-dispatch-ab.py dispatch.json
```

`raw/gguf-barrier-worker-0001.sh` is the exact worker that produced `raw/dispatch.json`.

## Correctness

Both implementations are held to one parameterized suite,
`GgufRowExecutorContractTest` (15 methods x 2 implementations = 30 tests), covering repeated
dispatches, ordered stage coverage, worker and stage failure propagation, concurrent publishers,
nested dispatch within and across pools, close-from-callback rejection, the maximum chunk
configuration, a zero poll budget, an oversubscribed 12-worker pool, and a soak that forces every
party through park-and-wake on every dispatch. The full `vectors-core` suite is 2017 tests, 0
failures.
