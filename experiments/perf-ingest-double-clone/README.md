# perf(db): eliminate double vector clone on ingest

## Hypothesis
When normalizeForCosine is true, ingest clones the vector twice:
1. maybeNormalizeForCosine() clones before normalising
2. StagingBuffer.append() clones again for defensive isolation

For normalizeForCosine=false, only StagingBuffer.append() clones.

Moving the clone responsibility to the caller (maybeNormalizeForCosine) and
removing it from StagingBuffer reduces the clone cost from 2 → 1 when
cosine normalisation is enabled, with no change when disabled.

## Decision rule
If all ingest and staging tests pass and no caller mutates a vector after
passing it through staging, accept.

## Kill criterion
If memory-safety tests fail (e.g. caller mutation leaks into stored data), reject.

## Implementation
- maybeNormalizeForCosine: clone the vector unconditionally (not just when
  normalizeForCosine is true)
- StagingBuffer.append: remove the defensive vector.clone() and document that
  the caller must provide an isolated Document
