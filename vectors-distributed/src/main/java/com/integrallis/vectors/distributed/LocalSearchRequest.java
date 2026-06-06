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
import java.util.Arrays;
import java.util.Objects;

/**
 * A search request destined for a single cluster node.
 *
 * <p>Carries the <b>full</b> {@link SearchRequest} so the node executes the caller's exact query —
 * metadata {@code filter}, projection flags ({@code includeVector/Text/Metadata}), and graph-search
 * tuning ({@code searchListSize}, {@code overQueryFactor}) all reach the node. (Previously this
 * record kept only {@code query/k/minScore}, which silently dropped the filter on scatter-gather —
 * a correctness bug for filtered distributed search.)
 *
 * <p>{@code clusterIds} is the subset of IVF cluster identifiers that the target node owns and
 * should search. An empty array signals a broadcast search (search all clusters on the node).
 *
 * @param targetNode the node that should execute this request
 * @param clusterIds cluster ids this node should restrict its search to (empty = search all)
 * @param request the full search request to execute on the node (query stored by reference)
 */
public record LocalSearchRequest(NodeId targetNode, int[] clusterIds, SearchRequest request) {

  public LocalSearchRequest {
    Objects.requireNonNull(targetNode, "targetNode must not be null");
    Objects.requireNonNull(clusterIds, "clusterIds must not be null");
    Objects.requireNonNull(request, "request must not be null");
  }

  /**
   * Builds a request from raw {@code query/k/minScore} (no filter, default projection) — the legacy
   * shape used by the IVF cluster router and tests. Prefer the canonical constructor with a full
   * {@link SearchRequest} when a filter or projection flags must be honoured.
   */
  public static LocalSearchRequest of(
      NodeId targetNode, float[] query, int[] clusterIds, int k, float minScore) {
    return new LocalSearchRequest(
        targetNode, clusterIds, SearchRequest.builder(query, k).minScore(minScore).build());
  }

  /** The query vector (delegates to {@link #request()}). */
  public float[] query() {
    return request.query();
  }

  /** Number of nearest neighbours requested (delegates to {@link #request()}). */
  public int k() {
    return request.k();
  }

  /** Minimum similarity threshold (delegates to {@link #request()}). */
  public float minScore() {
    return request.minScore();
  }

  @Override
  public String toString() {
    return "LocalSearchRequest[targetNode="
        + targetNode
        + ", k="
        + request.k()
        + ", clusterIds="
        + Arrays.toString(clusterIds)
        + ", minScore="
        + request.minScore()
        + ", filtered="
        + (request.filter() != null)
        + "]";
  }
}
