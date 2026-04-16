package com.integrallis.vectors.distributed;

import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.util.Objects;

/**
 * In-process {@link NodeSearchClient} backed by a real {@link VectorCollection}.
 *
 * <p>Used in tests to simulate a distributed cluster without networking. Each instance wraps one
 * local {@link VectorCollection}; the {@link LocalSearchRequest}'s {@code clusterIds} field is
 * ignored — the full collection is searched (appropriate for FLAT / HNSW nodes in tests).
 */
final class InProcessNodeSearchClient implements NodeSearchClient {

  private final VectorCollection collection;

  InProcessNodeSearchClient(VectorCollection collection) {
    this.collection = Objects.requireNonNull(collection, "collection must not be null");
  }

  @Override
  public SearchResult search(LocalSearchRequest request) {
    // clusterIds is informational for IVF routing; for in-process FLAT/HNSW collections
    // we forward the full query and let the collection search all its data.
    SearchRequest sr =
        SearchRequest.builder(request.query(), request.k()).minScore(request.minScore()).build();
    return collection.search(sr);
  }

  @Override
  public int size() {
    return collection.size();
  }

  @Override
  public int physicalSize() {
    return collection.physicalSize();
  }
}
