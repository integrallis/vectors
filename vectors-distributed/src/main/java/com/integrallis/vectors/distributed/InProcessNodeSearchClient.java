/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
package com.integrallis.vectors.distributed;

import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.util.Objects;

/**
 * In-process {@link NodeSearchClient} backed by a real {@link VectorCollection}.
 *
 * <p>Simulates a distributed cluster node within a single JVM — no network required. Each instance
 * wraps one local {@link VectorCollection}; the {@link LocalSearchRequest}'s {@code clusterIds}
 * field is informational and ignored (appropriate for FLAT / HNSW nodes). This is the primary
 * integration vehicle for unit tests and JMH benchmarks.
 */
public final class InProcessNodeSearchClient implements NodeSearchClient {

  private final VectorCollection collection;

  /**
   * @param collection the backing collection for this simulated node (must not be null)
   */
  public InProcessNodeSearchClient(VectorCollection collection) {
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
