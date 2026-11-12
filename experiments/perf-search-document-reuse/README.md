# perf(db): reuse stored Document on search when projection is identical

## Hypothesis
VectorCollectionImpl.search() creates a new Document per hit via:
```java
new Document(stored.id(), vector, includeText ? stored.text() : null, ...)
```
For the common case where the request includes all fields and the document is
already fully hydrated (in-memory mode), this Document is identical to `stored`
and the allocation is wasted.

## Decision rule
If search-path tests pass, accept.

## Kill criterion
If any test fails because a returned document is missing fields that were
requested, reject.

## Implementation
Add a `projectDocument(stored, vector, request)` helper that returns `stored`
directly when:
1. `vector == stored.vector()` (no hydration needed)
2. The request includes all non-null fields in `stored`

Otherwise it allocates a new Document with the projected fields.

Two sites updated:
- `search()` — the main search path
- `searchFlatBatch()` — the batch search path
