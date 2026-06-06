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

import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
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
  private final String requiredBearerToken;

  /**
   * @param collection the backing collection for this simulated node (must not be null)
   */
  public InProcessNodeSearchClient(VectorCollection collection) {
    this(collection, null);
  }

  /**
   * @param collection the backing collection for this simulated node (must not be null)
   * @param requiredBearerToken bearer token required for node-to-node calls; {@code null} disables
   *     authentication
   */
  public InProcessNodeSearchClient(VectorCollection collection, String requiredBearerToken) {
    this.collection = Objects.requireNonNull(collection, "collection must not be null");
    this.requiredBearerToken = requiredBearerToken;
  }

  @Override
  public SearchResult search(LocalSearchRequest request) {
    return search(request, NodeCallContext.none());
  }

  @Override
  public SearchResult search(LocalSearchRequest request, NodeCallContext context) {
    requireAuthorized(context);
    // clusterIds is informational for IVF routing; for in-process FLAT/HNSW collections we run the
    // node's full SearchRequest verbatim so the caller's filter and projection flags are honoured.
    return collection.search(request.request());
  }

  @Override
  public int size() {
    return size(NodeCallContext.none());
  }

  @Override
  public int size(NodeCallContext context) {
    requireAuthorized(context);
    return collection.size();
  }

  @Override
  public int physicalSize() {
    return physicalSize(NodeCallContext.none());
  }

  @Override
  public int physicalSize(NodeCallContext context) {
    requireAuthorized(context);
    return collection.physicalSize();
  }

  private void requireAuthorized(NodeCallContext context) {
    if (requiredBearerToken == null) {
      return;
    }
    String actual = context == null ? null : context.bearerToken();
    if (actual == null
        || !MessageDigest.isEqual(
            actual.getBytes(StandardCharsets.UTF_8),
            requiredBearerToken.getBytes(StandardCharsets.UTF_8))) {
      throw new SecurityException("missing or invalid node bearer token");
    }
  }
}
