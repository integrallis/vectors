# perf(db): skip BitSet.clone() on append-only commits

## Hypothesis
commitInMemory() and commitPersistent() always clone the old generation's
tombstones BitSet, even when the commit is append-only (no pending tombstones).
For a 10M-document collection this copies ~1.2MB of data per commit that is
immediately discarded unchanged.

The common workload is append-only ingestion. For this case we can share the
predecessor's tombstones BitSet by reference, since Generation is immutable and
the BitSet is never mutated after construction.

## Decision rule
If all commit-path tests pass, accept.

## Kill criterion
If any test fails because a tombstone from a later generation leaks into an
earlier one, reject.

## Implementation
- Extract `staging.pendingTombstones()` into a local variable
- When the set is empty, assign `oldGen.tombstones` directly to `newTombstones`
- When the set is non-empty, clone as before and apply tombstones
