# Qualification status — 2026-10-03

Final candidate measurements are running. This commit makes no qualified collection speedup claim.
The baseline is `4bc505b3e8c0c217e9d66727d643921a0d8efd55` (including the prior batch-write fix).

The replacement applies automatically to exact construction. SQ8 is excluded. Rejected variants
and their results are described in [README.md](README.md); their speedups are not shipping claims.
Results, raw samples and source provenance will be attached before the PR is made ready to merge.

The tests were introduced before the corresponding fixes: selected red/green logs are retained in
`evidence/`. Those historical logs also include subsequently rejected worker/segment experiments;
only the final source and final CI run define what is being delivered.
