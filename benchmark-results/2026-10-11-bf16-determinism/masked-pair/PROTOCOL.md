# BF16 final reduction comparison

Retain the failed rotate/slice experiments and the complete final-expressions results. Test the
published baseline, original fixed tree, SVE lane formulation and masked final-pair formulation
under six counterbalanced process pairs on each dedicated x86/ARM host, all six existing
kernel/shape combinations. Every candidate first passes all twelve cold/compiled probes and the
allocation check. The new mask is an engineering hypothesis, not an accepted production route.

Keep five one-second warmups and five one-second measured iterations per fork and the GC profiler.
Use each process pair as the independent unit; compare paired latency ratios with confidence
intervals, retaining all runs. Investigate every detected slowdown; do not promote a candidate
merely because it fits the earlier five-percent investigation threshold. Follow with complete
real-weight model checks and full builds before publication. No runtime architecture branch is
justified unless that candidate wins on the affected hardware and preserves other cases.

Use the pinned JDK from the DeltaNet campaign, one-hour host deadlines and a two-dollar combined
ceiling. Retain artifact/source hashes and observed instance termination. This is a component
experiment and cannot substitute for the two-public-dataset model qualification.
