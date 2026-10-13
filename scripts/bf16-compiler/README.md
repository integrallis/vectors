# BF16 compiler diagnosis

One Neoverse-N2 host runs 32 matched ordinary-matvec pairs. One x86 runner runs 32
small-matvec pairs only if it identifies as EPYC 9V45 before timing. An unmatched
runner retains its hardware receipt and performs no measurement; that CPU scope
remains uncovered. No other CPU is silently substituted.

The unchanged public JMH benchmark uses five one-second warmup iterations and five
one-second measurement iterations in each fresh JVM, with counterbalanced pairs.
Every process retains compilation XML, printed BF16 generated code, raw iteration
scores and allocation metrics. All 32 pairs remain, regardless of mode or speed.
The instrumented results diagnose compilation; they cannot pass a release gate.
Modes must be identified from compiler/code metadata, never excluded using a time
cutoff. No runtime implementation changes or compiler overrides select a faster path.
