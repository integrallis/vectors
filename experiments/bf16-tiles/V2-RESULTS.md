# Two-query BF16 tile results

Measured source `2a3d575`, payload
`4a27eaec0c492054456e1b4ed9aec66f12e3a8847d55d772b76b38582f3f12ad`,
six complete counterbalanced process pairs on one m6a.4xlarge and one m7g.4xlarge.
All twelve cold/hot probes and the existing allocation gate passed on each host.
No performance conclusion applies to later production changes until their rerun.

All x86 comparisons and five ARM comparisons have confidence intervals below one.
The ARM small matvec ratio is 0.99634, 95% interval [0.98706, 1.00326]:
**inconclusive**, and not a no-regression pass. Precision targets were not met for
x86 tall matvec or ARM medium/tall batching. Keep every pair in the adjacent analysis.

The allocation probe measured 102,992 bytes over 1,000 thirteen-token calls on both
hosts (102.992 bytes/call), below the existing 256-byte/call gate. Medium matvec
allocation decreased from approximately 233,476 bytes/op to 41.4–41.6 bytes/op.
These are allocations measured here; they do not establish zero allocation.

The production branch adds nested-executor protection, source cleanup and direct
ordinary-row reductions. Final combined-artifact kernel and actual-model evidence
will use a separately frozen twelve-pair kernel protocol.
