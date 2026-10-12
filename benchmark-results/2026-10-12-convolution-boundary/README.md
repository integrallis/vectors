# Convolution boundary: cold regression persists

This isolated candidate is **not qualified for adoption**. It leaves a material cold allocation
regression and does not establish a warmed latency improvement over the broad candidate. No
production branch, publication, CI job or cloud host was changed.

Actual Gradle JARs were measured in six rotating blocks of published/broad/candidate processes
on the recorded local x86_64 Temurin25.0.3 host. Every process uses10240channels, four taps,
1000changing-input warmups and500measured calls with consumed output. First-call costs include
activation/provider class initialization. These component timings are diagnostic; they are not
whole-model or supported-host qualification, and the host is a shared development machine.

| Metric | Candidate / broad, paired95%CI | Candidate / published, paired95%CI |
| --- | --- | --- |
| First-call allocated bytes |0.9784 [0.9767,0.9796]|4.2186 [4.2109,4.2233]|
| First-call elapsed time |0.9207 [0.8421,1.0113]|2.8895 [2.7286,3.0334]|
| Warmed elapsed time |0.9621 [0.9020,1.0286]|0.5872 [0.5664,0.6128]|

Broad and candidate allocate zero warmed bytes in all six processes; published allocates143360
bytes per call. Broad cold allocation is about22MB; candidate about21.5MB; published about5.1MB.
An array-only boundary with identical inline exponential arithmetic therefore does not remove
the cold vector-object cost. It must not be described as fixing that regression.

All18processes match independent scalar replay for every cold/hot output and persistent history
cell, preserving the selected fused-vector/unfused-tail arithmetic. The focused Gradle run
passes84tests. Separate fresh-JVM tests cover the48-head sigmoid/softplus contract,259-channel
four-tap persistent sequences, all width ceilings and disabled FMA; exact logs and commands are
retained. The512ceiling resolves256on this host and is not native512coverage.

The candidate source is aabe00845f0e81828b7fc4992ce7515be4459d88. Artifact hashes and source
revisions are in manifest.json; raw process results and paired analysis are retained unchanged.
The first build failure (a duplicate local variable introduced while copying the arithmetic)
and corrected successful build log are both preserved.
