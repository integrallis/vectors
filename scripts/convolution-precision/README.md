# Convolution precision cohort

This prospective cohort measures the unchanged public convolution operation with pinned candidate and published artifacts. It does not pool the earlier four-pair screening cohorts. Required CPU strata are EPYC7763, EPYC9V74 and Neoverse-N2; no aggregate can compensate for a failing or missing stratum.

Hardware admission precedes builds or measurements: eight fixed attempts each for7763/9V74 and one N2 runner. Every matching host completes32 fresh-process pairs, with16 AB and16 BA orders. All matching hosts and forks remain in analysis; mismatches retain metadata only. There is no first-pair performance stopping rule in this cohort. Setup or numerical failures retain partial receipts and prevent a complete verdict.

Each process performs the original10000 warmup calls, then five blocks containing at least one second of accumulated public-call time. Every block contains a multiple of the complete17-input cycle. Input preparation and allocation-counter reads remain outside the measured call. All five block times, counts and allocation totals are retained. The process estimate is total measured call time divided by total measured calls.

Analysis uses equal-host means of paired log ratios separately for each CPU, with20000 hierarchical bootstrap draws and fixed seed716309. The original upper-ratio1.0, interval-width0.02 and allocation limits remain unchanged. Every required stratum must pass. A stratum represented by one host supports only that measured host; the report states host counts. Even a passing component result does not qualify whole-model inference or public QA.
