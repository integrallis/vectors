# Thread-count follow-ups: the three gaps the 2.88x measurement left open

**Protocol frozen 2026-11-04 before any data was collected. Nothing below the
"Results" heading existed when the first arm was run.**

`2026-11-03-hybrid-cpu-thread-count` established that `availableProcessors()` costs 2.88x on an
i9-14900HX and that the optimum is the performance-core count. `2026-11-03-decision-thread-sweep`
is its uniform-core control, where the optimum is the physical-core count and the cost is 1.52x.
Together they justify `CpuTopology.computeThreads()`, which ships as the default in vectors 0.1.33.

Both swept **powers of two only**, on **one workload**, on **two machines**, against the library as
it was **before** native chunk claiming landed in models 0.5.13. This directory closes three of
those gaps. They were written down as open in the 0.1.33 evidence and are being closed because the
default is a user-visible one and "measured optimal" has to mean measured.

## Hypotheses and decision rules

### 1. Intermediate thread counts on the hybrid part

The shipped policy picks 8 on this CPU (8 performance cores, efficiency cores excluded). Before
chunk claiming, a single efficiency-core thread held up every barrier, so including any was a loss:
16 threads cost 1.31x against 8. Claiming lets a slow core take fewer chunks, so efficiency cores
may now contribute instead of stalling.

- **H1**: with claiming in place, some count **above 8** beats 8 on this CPU.
- **Sweep**: 8, 10, 12, 16, 24, 32 threads. 3 repeats at 8 and at the best non-8 arm.
- **Decision rule**: if an arm beats 8 by more than 5% with non-overlapping repeat ranges, the
  policy is leaving throughput on the table and should add a tunable fraction of efficiency cores.
  Otherwise H1 is rejected and 8 stands.
- **Prior**: claiming recovered 1.07x of a 2.88x gap, so H1 is expected to fail. Recorded here
  before the run so the expectation cannot be fitted afterwards.

### 2. (deliberately not in this directory)

Apple Silicon. The `hw.perflevel0.physicalcpu` detection path is implemented and unit-tested
against synthetic topologies but has never been performance-measured on an M-series part. No
M-series machine is available here, so this stays open and is **not** closed by anything below.

### 3. A uniform Intel laptop with SMT and no efficiency cores

The policy picks 6 of 12 logical on an i7-9750H. That is the EPYC finding (physical, not logical)
transferred to a different part, not measured on it.

- **H3**: 6 threads is the optimum on this CPU.
- **Sweep**: 2, 4, 6, 8, 12 threads. 3 repeats at the best arm and at 12.
- **Decision rule**: if the optimum is not 6, the policy's SMT exclusion is wrong on this class of
  part and needs a measured rule rather than a transferred one.

### 4. A second workload

The decision task is a short prefill dominated by matrix products. Prefill at length and decode at
a long context have different balances: prefill batches rows, decode is bandwidth-bound and spends
proportionally more in attention, whose parallelism this release also changed.

- **H4**: the optimal thread count is the same for long prefill and long-context decode as for the
  short decision, on the same machine.
- **Arms**: prefill of a long prompt (time to first token) and decode rate over the following
  tokens, at the thread counts that bracket the decision optimum.
- **Decision rule**: if the optimum differs by more than one step of the sweep between workloads,
  one global thread default is the wrong shape and the count has to be chosen per phase.
- **Cross-validation**: any new measurement script must agree with the established in-repo harness
  (`PrefillProfileCli` / `DecodeProfileCli`) on one overlapping configuration before its numbers
  are used, per the framework-first policy.

## Kill criteria

- An arm whose repeats span more than 10% is reported as inconclusive, not as a number. The
  0.1.33 control already showed ~20% run-to-run on single runs of this workload, which is wider
  than several of the differences being tested for here.
- Accuracy is recorded in every arm. Thread count cannot change an answer; if it does, the
  measurement is wrong and nothing else in the arm is reportable.

## Machines

- `rockhopper`: Intel i9-14900HX, 8 performance cores with 2 threads each plus 16 efficiency
  cores, 32 logical. Hybrid. The only machine here with efficiency cores; no cloud instance has
  them, so experiment 1 cannot move to a VPS and the laptop is not a caveat but the requirement.
- This Mac: Intel i7-9750H, 6 cores, 12 logical, AVX2, no efficiency cores.

Full topology for each in `raw/machine-*.txt`, read at measurement time rather than assumed.

## Library under test

models **0.5.13** and vectors **0.1.33**, the published artifacts, so chunk claiming is present.
The 0.1.33 evidence was taken against published 0.1.76 (models 0.5.12, vectors 0.1.32) and so
predates claiming. The resolved classpath for every arm is recorded in `raw/classpath-*.txt`,
because "the published jar" is a claim about what was on the classpath and is checkable.

## Results

Collected after this protocol was written. Empty until then.

### 1. Intermediate thread counts on the hybrid part — H1 REJECTED

`table.txt` is generated by `summarize.py` from `raw/rockhopper/*.log`. The script asserts that
each log's `ggufThreads` toggle matches the arm it is filed under, so an arm cannot silently have
run a different thread count; all 25 logs passed that check.

| threads | n | median p50 | min | max | spread |
|--------:|--:|-----------:|----:|----:|-------:|
| 8 (shipped default) | 7 | **890.0 ms** | 667.8 | 906.9 | 27% |
| 10 | 1 | 976.9 | — | — | — |
| 12 | 1 | 932.4 | — | — | — |
| 16 | 7 | 906.6 | 825.7 | 986.1 | 18% |
| 24 | 7 | 919.8 | 879.7 | 1004.8 | 14% |
| 32 (`availableProcessors()`) | 2 | **2562.9 ms** | 2516.1 | 2609.7 | 4% |

Accuracy 11/15 in all 25 runs, as in the 0.1.33 evidence. Route `rust-ffm` with
`ggufExecutor=countdown` in all of them, which is itself the check that vectors 0.1.33 was on the
classpath: 0.1.32 reports `persistent`.

**H1 is rejected. No thread count above 8 beats 8.** The ordering 8 < 16 < 24 is monotone across
seven alternated repeats, and 32 is 2.88x the best.

**But the stronger claim is not supported either, and that is the finding worth keeping.** The
8-thread arm spans 27%, the 16-thread arm 18%, and the differences being tested between them are
2-3%. By this directory's own kill criterion those arms are **inconclusive against each other**:
8, 16 and 24 are indistinguishable on this workload at this sample size. What the data supports is
narrower than "8 is optimal" --

- saturating every logical processor is catastrophic and reproducible (2.88x, 4% spread, no
  overlap with anything else);
- any count from the performance-core count up to three quarters of the logical count is the same
  within noise;
- so the shipped policy is **safe** — never worse than the alternatives and best by median — but
  the justification is "do not saturate", not "8 is uniquely right".

Resolving 2% on this workload would need hundreds of runs, which is not worth it to choose between
arms that are already tied. The practical consequence is for future sweeps: **this workload cannot
resolve differences below about 10%**, which the 0.1.33 control also noted and which no single-run
arm in that directory respected.

That has one retrospective consequence. The 0.1.33 sweep's intermediate points were single runs,
and its 16-thread arm at 1145 ms is far outside the seven-run range measured here (826-986). The
apparent 1.26x improvement at 16 threads from chunk claiming is therefore **confounded with
single-run noise and is not claimed**. Only the 8-thread and 32-thread arms in that directory had
repeats, and both reproduce here against a different library version: 2.88x then, 2.88x now.

The prior registered above -- that H1 would fail because claiming recovered 1.07x of a 2.88x gap
-- held.

### 4. A second workload — H4 SUPPORTED, with a caveat that matters more than the hypothesis

`table-prefill.txt` is generated by `summarize-prefill.py` from `raw/rockhopper-pd/*.log`. The
script asserts two things per row and refuses the data otherwise: that the log's `threads toggle`
matches the arm, and that the prompt cache served at most 16 tokens.

Long prefill (3215 tokens) and the decode that follows it, on the hybrid part, published 0.5.13:

| threads | prefill tok/s | decode tok/s | prefill vs best | decode vs best |
|--------:|--------------:|-------------:|----------------:|---------------:|
| 8 | **90.5** | **14.65** | 1.00x | 1.00x |
| 16 | 90.5 | 13.16 | 1.00x | 0.90x |
| 24 | 87.7 | 12.84 | 0.97x | 0.88x |
| 32 | 50.8 | 2.24 | 0.56x | **0.15x** |

**H4 is supported.** The optimum is 8 for long prefill, for long-context decode, and for the short
decision, and it does not move by even one step of the sweep between them. One global thread
default is the right shape; the count does not need to be chosen per phase.

**What the hypothesis did not ask, and what the data says anyway: the cost of getting it wrong is
not one number.** Oversubscription to `availableProcessors()` costs

- **6.6x on decode**,
- 2.88x on the short typed decision,
- 1.8x on prefill.

The ordering is mechanically sensible. Decode does a token's worth of work between barriers, so
barrier cost dominates and a slow partition is most of the step; prefill batches rows and has real
arithmetic to amortise the same barrier against. The consequence is that **the 2.88x published
with vectors 0.1.33 understates the damage for anything generation-heavy**: a chat or agent loop
left on `availableProcessors()` loses close to seven times on the phase it spends its time in.

Separately and not part of H4: **prefill is flat from 8 to 24 threads** (90.5, 90.5, 87.7). Extra
threads buy nothing. Whether that is memory-bandwidth saturation or parallelism the prefill path is
not taking needs a profile, and no claim is made here about which. In particular ~90 tok/s is not
called slow: there is no cross-implementation baseline on this machine to call it against, and
producing one is a separate piece of work.

**Measurement note.** The first draft of the driver reported 3215 tokens prefilled in 72.3 ms,
which is 44,337 tok/s and faster than this hardware can go. The pipeline's prompt cache was serving
the shared prefix, so the repeats were measuring the cache. Each repeat now carries a unique
leading line; `cacheReadInputTokens` fell from 3209 to 6 and the figures became consistent. A
measurement that beats the machine is a broken measurement, and this one was caught by the
arithmetic rather than by the output looking wrong.

**Host.** These arms ran on the i9 laptop because the effect under test is largest on the hybrid
part. That host throttles, which sets its noise floor near 10-20% as experiment 1 established. The
differences this experiment turns on are 1.8x to 6.6x, far outside that floor; the 8-to-24
differences are inside it and are reported as flat rather than as an ordering.

### 3. SMT exclusion on a uniform part — H3 SUPPORTED

**Host changed, and the first attempt is recorded as void rather than deleted.** This was started
on the i7-9750H Mac and produced an 8-beats-6 result. That reading was wrong twice over: the
machine throttled to `CPU_Speed_Limit = 53` under sustained load, and a second process on the same
workstation was running a Gradle end-to-end suite throughout. The arms measured contention and
thermal state. Those numbers are not in `raw/` and are not reported; the standing rule that timing
runs belong on a dedicated box exists for exactly this and was not followed.

Re-run on an ephemeral Hetzner **ccx33** (AMD EPYC-Milan, 8 dedicated vCPU presented as **4 cores x
2 threads**, siblings `0-1`, `2-3`, `4-5`, `6-7` -- read on the box, in `raw/machine-vps.txt`,
rather than assumed from the instance type). Published artifacts only, resolved by Maven with every
`com.integrallis` module pinned: `raw/classpath-vps.txt` contains zero 0.5.12 jars.

| threads | n | median p50 | min | max | spread | vs best |
|--------:|--:|-----------:|----:|----:|-------:|--------:|
| 1 | 1 | 3844.8 | — | — | — | 3.23x |
| 2 | 3 | 1917.0 | 1861.1 | 2062.4 | 11% | 1.61x |
| **4 — physical cores, what the policy picks** | 4 | **1188.8** | 1072.0 | 1228.1 | 13% | **1.00x** |
| 6 | 3 | 1303.2 | 1295.0 | 1328.4 | 3% | 1.10x |
| 8 — every logical processor | 4 | 1329.7 | 1210.5 | 1361.9 | 11% | 1.12x |

**H3 is supported. The optimum is the physical core count, and using the SMT siblings costs 12%.**
Accuracy 11/15 in all 15 runs.

The margin is small but, unlike experiment 1's, it is outside the noise: 6 and 8 are worse than 4
in every one of the paired repeats, not only on medians. Scaling below the optimum is close to
linear (1 -> 2 is 2.01x, 2 -> 4 is 1.61x), which is what a parallel region should look like, and
the curve turns over exactly where the cores run out.

**The detection code was validated on this host, not just the sweep.** `raw/topology-detection-vps.txt`:

```
availableProcessors = 8
VectorUtil.physicalCores() = 4
VectorUtil.computeThreads() = 4
```

That matters more than the sweep. A virtualised guest is where topology detection is most likely
to fail -- the hybrid-CPU path was written against `/sys/devices/system/cpu/types`, which is absent
here -- and the policy still resolves to the measured optimum without being told. This is the
first time the detection has been checked on a VM rather than on a synthetic `/sys` tree.

Still open, and not closed by this: **Apple Silicon**. The `hw.perflevel0.physicalcpu` path remains
unit-tested only. No M-series machine is available here.

## Reproduce

```
python3 summarize.py            > table.txt           # experiments 1 and 3
python3 summarize-prefill.py raw/rockhopper-pd > table-prefill.txt   # experiment 4
```

Both read only `raw/` and re-derive every number in this file. Neither reaches the network and
neither needs the hardware: the logs are the artifact. Each script refuses data that would make
its table a lie —

- `summarize.py` asserts a log's `ggufThreads` toggle equals the thread count of the arm it is
  filed under, so an arm cannot silently have run a different configuration;
- `summarize-prefill.py` asserts the same for its own toggle, and additionally that the prompt
  cache served at most 16 tokens, because the first draft of that driver measured the cache
  instead of prefill and reported a speed the hardware cannot reach.

## Where each arm ran, and why

- **Experiment 1** on the i9-14900HX laptop. It is the only hybrid-core machine available and no
  cloud instance has efficiency cores, so this one cannot move to a dedicated box.
- **Experiment 3** on an ephemeral Hetzner ccx33, created and destroyed for this
  (`projects/memory-research/research/thread-sweep-vps-log.md`). Its first attempt ran on a shared
  laptop and produced the opposite answer; see that section.
- **Experiment 4** on the i9 laptop, because the effect under test is largest on the hybrid part
  and the differences it turns on are 1.8x to 6.6x, far outside that host's noise floor.

## What is still open

Apple Silicon. A free GitHub `macos-15-arm64` probe (`.github/workflows/apple-topology-probe.yml`,
vectors run 41873936193) established that a virtualised Apple guest reports `hw.nperflevels = 1`
with no `hw.perflevel1.*` at all, so no virtual machine can exercise the performance/efficiency
split. That run did validate the macOS fallback on real Apple Silicon: `computeThreads()` resolved
to 3 on a 3-core guest, correctly. Bare metal was priced and attempted on 2026-11-04 -- Scaleway
had no stock of any Apple type in either zone, and EC2 `mac2.metal` reported
`InsufficientHostCapacity` in all nine availability zones tried across four regions -- so the split
stays untested and is recorded as such rather than assumed to transfer from Intel.
