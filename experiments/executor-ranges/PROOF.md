# Persistent executor integer-limit proof

The frozen ae337eeb artifact can emit row -2147483648 for a valid positive row count of
Integer.MAX_VALUE. Its stage route can silently execute no ranges, and requesting the maximum
chunk configuration fails the bounded five-second child-process check before any callback.
The probes stop after the first valid row or validate a small number of whole stage ranges;
they never iterate two billion rows or allocate a matrix of that size.

The fix uses `1 + (count - 1) / chunks` for positive counts and chunk counts, performs start/end
arithmetic in long, and rejects a wrapped negative chunk counter. Since both factors are at most
Integer.MAX_VALUE, their product and the following addition fit in long. Clamping the end to the
positive int row count happens before narrowing; valid ranges therefore stay in the int domain.
For ordinary values where the old sum/product fit in int, these expressions are algebraically
identical to the old calculation.

That equivalence was also checked by actual dispatch. Seventy combinations of seven row counts,
two pool sizes and five chunk settings produced identical sorted ranges on the old and fixed
artifacts: SHA256 6f0383c11fe9a0dc55c3d20c4a40282716ce5715f0b7011b9e4684b17340c9d4.
Counts include small irregular matrices, 4864 and 151936 rows. Maximum-stage probes validate
complete, contiguous, non-overlapping coverage up to Integer.MAX_VALUE in bounded time.
All three failing child cases pass after the fix. This establishes correctness and ordinary
partition equivalence; it is not a latency measurement.

The frozen full-model campaign artifact is untouched. This isolated change has separate source,
artifact and test receipts and must be integrated deliberately.
