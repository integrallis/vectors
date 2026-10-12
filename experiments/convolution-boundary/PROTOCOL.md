# Convolution boundary experiment

Local component diagnostic only; not production qualification. The actual published core and
broad source6893a587 JARs first ran two alternating process pairs on the same host. The broad
route allocated zero warmed bytes but approximately22MB on the first call, versus approximately
5.1MB published. All full outputs and persistent history matched independent scalar replay of
the selected fused-vector/unfused-tail arithmetic contract.

The candidate only moves convolution into an array-only helper and inlines the existing double
exponential computation there, analogous to the existing SwiGLU boundary. No approximation,
FMA decision, scalar-tail boundary or history ordering changes.

Before measuring the candidate, freeze six blocks of three independent processes, rotating
published/broad/candidate order. Every process uses10240channels, four taps, changing input,
1000warmup calls and500measured calls; outputs are consumed. Record first-call allocated bytes
and elapsed time separately from warmed allocated bytes/latency. The cold call includes actual
activation class initialization. Retain every result and full cold/hot output/history oracle
mismatch count. Compile one harness against the published artifact and use it unchanged for all
three arms. Use actual Gradle core JARs, recording hashes and source provenance.

This experiment can reject a boundary proposal or justify further whole-model testing. It
cannot establish portability or clear existing production memory/startup gates. Report all
ratios with paired bootstrap intervals; six local blocks do not waive precision requirements.
A cold-allocation reduction that retains a material regression against published remains an
unresolved regression, not a release pass.
