# perf(db): eliminate List.copyOf allocation in SearchResult constructor

## Hypothesis
SearchResult's canonical constructor calls List.copyOf(hits), which allocates a
new Object[k] array and copies all Hit references for every search call. For k=100
results, this is 400 bytes of transient allocation per search. The caller
(VectorCollectionImpl) never mutates the ArrayList after construction, so the
copy is pure overhead.

Replacing List.copyOf with Collections.unmodifiableList reuses the existing
ArrayList instead of copying.

## Decision rule
If all search-path tests pass and no mutable list leaks into a SearchResult, accept.

## Kill criterion
If a SearchResult is ever observed to have a mutable list backing it, reject.

## Implementation
Replace `hits = List.copyOf(hits)` with `hits = Collections.unmodifiableList(hits)`
in the SearchResult canonical constructor.
