# perf(db): normalize vectors outside the write lock on ingest

## Hypothesis
VectorCollectionImpl.add() and addAll() call maybeNormalizeForCosine() inside
stageUnderLock(), which holds the write lock. Vector normalization is O(dim)
work (dot product, sqrt, division). For batch ingestion with normalizeForCosine=true,
this serializes all normalization work under the lock, blocking concurrent readers.

Moving normalization outside the lock allows readers to proceed during the CPU-
bound normalization phase.

## Decision rule
If all add/addAll tests pass, accept.

## Kill criterion
If any test fails because a normalized vector is not visible in the committed
generation, reject.

## Implementation
1. In add(): call maybeNormalizeForCosine() before acquiring the write lock
2. In addAll(): normalize all documents before acquiring the lock
3. In stageUnderLock(): remove maybeNormalizeForCosine() call; doc is already normalized
