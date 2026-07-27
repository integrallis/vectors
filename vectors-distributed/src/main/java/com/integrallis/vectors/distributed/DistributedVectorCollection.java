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

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.filter.Filter;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.VectorCollectionConfig;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Implements the full {@link VectorCollection} API across a distributed cluster.
 *
 * <p><b>Write path</b>: {@code add}, {@code upsert}, {@code delete}, {@code commit}, and {@code
 * compact} are delegated to the <i>local</i> node's collection. In Phase 3, a single node acts as
 * the write primary; all mutations target that node.
 *
 * <p><b>Read path</b>: {@code search} broadcasts to all nodes via {@link ScatterGatherExecutor} and
 * merges results. {@code get} and {@code contains} are forwarded to the local node only (no
 * cross-node lookup in Phase 3).
 *
 * <p><b>Consistency</b>: read-your-own-writes within a session; eventual consistency across nodes.
 *
 * <p>Use {@link Builder} to construct an instance.
 */
public final class DistributedVectorCollection implements VectorCollection {

  private final VectorCollection localCollection;
  private final ScatterGatherExecutor executor;
  private final List<NodeId> allNodes;
  private final NodeDirectory directory;
  private final NodeId localNodeId;
  private final NodeCallContext nodeCallContext;
  private final Duration timeout;

  private DistributedVectorCollection(
      VectorCollection localCollection,
      NodeId localNodeId,
      NodeDirectory directory,
      List<NodeId> allNodes,
      Duration timeout,
      NodeCallContext nodeCallContext) {
    this.localCollection = localCollection;
    this.localNodeId = localNodeId;
    this.directory = directory;
    this.allNodes = List.copyOf(allNodes);
    this.nodeCallContext = nodeCallContext;
    this.timeout = timeout;
    this.executor = new ScatterGatherExecutor(directory, timeout, nodeCallContext);
  }

  // ---- Write path: delegate to local collection ----

  @Override
  public void add(Document doc) {
    localCollection.add(doc);
  }

  @Override
  public void addAll(Collection<Document> docs) {
    localCollection.addAll(docs);
  }

  @Override
  public void upsert(Document doc) {
    localCollection.upsert(doc);
  }

  @Override
  public boolean delete(String id) {
    return localCollection.delete(id);
  }

  @Override
  public int deleteWhere(Filter filter) {
    return localCollection.deleteWhere(filter);
  }

  @Override
  public void commit() {
    localCollection.commit();
  }

  @Override
  public void flush() {
    localCollection.flush();
  }

  @Override
  public void compact() {
    localCollection.compact();
  }

  // ---- Read path ----

  @Override
  public SearchResult search(SearchRequest request) {
    // Build a broadcast plan: one LocalSearchRequest per node, empty clusterIds = search all.
    // Carry the full SearchRequest so each node honours the caller's filter + projection flags.
    List<LocalSearchRequest> plan = new ArrayList<>(allNodes.size());
    for (NodeId node : allNodes) {
      plan.add(new LocalSearchRequest(node, new int[0], request));
    }
    return executor.execute(plan, request.k());
  }

  /** Looks up the document on the local node only (Phase 3: no cross-node get). */
  @Override
  public Document get(String id) {
    return localCollection.get(id);
  }

  /** Checks the local node only (Phase 3: no cross-node contains). */
  @Override
  public boolean contains(String id) {
    return localCollection.contains(id);
  }

  /**
   * Returns live documents from the <em>local</em> shard only.
   *
   * <p>For a full corpus export across all shards, call {@code documents()} on each node's {@link
   * VectorCollection} individually and merge the results.
   */
  @Override
  public List<Document> documents() {
    return localCollection.documents();
  }

  /**
   * Aggregates document counts across all nodes. Unlike the old sequential loop (no timeout, no
   * error handling — one down/slow node hung or failed the whole call), this fans out in parallel
   * under {@code timeout} and, if any node times out or throws, fails fast with a {@link
   * PartialResultException} carrying the partial sum and the unreachable nodes. The call is always
   * bounded by {@code timeout}.
   *
   * @throws PartialResultException if one or more nodes were unreachable within the timeout
   */
  @Override
  public int size() {
    return aggregate(c -> c.size(nodeCallContext), "size");
  }

  /**
   * Aggregates physical (tombstone-inclusive) document counts across all nodes. Same
   * fault-tolerant, timeout-bounded semantics as {@link #size()}.
   *
   * @throws PartialResultException if one or more nodes were unreachable within the timeout
   */
  @Override
  public int physicalSize() {
    return aggregate(c -> c.physicalSize(nodeCallContext), "physicalSize");
  }

  /**
   * Fans {@code call} out to every node in parallel, waits up to {@code timeout}, and sums the
   * successful responses. Timed-out or thrown responses are collected as unreachable nodes; if any
   * exist the whole aggregate throws {@link PartialResultException} (a partial <em>count</em> is
   * misleading, so we signal degradation rather than silently under-report) while still carrying
   * the partial sum and the missing nodes for callers willing to accept a partial answer.
   */
  private int aggregate(java.util.function.ToIntFunction<NodeSearchClient> call, String op) {
    List<java.util.concurrent.Callable<Integer>> tasks = new ArrayList<>(allNodes.size());
    for (NodeId node : allNodes) {
      tasks.add(() -> call.applyAsInt(directory.clientFor(node)));
    }
    int total = 0;
    java.util.Set<NodeId> missing = new java.util.LinkedHashSet<>();
    try (var vt = java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()) {
      List<java.util.concurrent.Future<Integer>> futures;
      try {
        futures =
            vt.invokeAll(tasks, timeout.toMillis(), java.util.concurrent.TimeUnit.MILLISECONDS);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new PartialResultException(
            op + " interrupted before completion", 0, java.util.Set.copyOf(allNodes));
      }
      for (int i = 0; i < futures.size(); i++) {
        NodeId node = allNodes.get(i);
        java.util.concurrent.Future<Integer> f = futures.get(i);
        if (f.isCancelled()) {
          missing.add(node); // timed out
          continue;
        }
        try {
          total += f.get();
        } catch (java.util.concurrent.ExecutionException e) {
          Throwable cause = e.getCause();
          // A security/auth failure is a systematic misconfiguration the caller must see (e.g. a
          // wrong node bearer token fails on every node), not a transient per-node availability
          // issue to mask as "unreachable". Propagate it (and any Error) directly.
          if (cause instanceof SecurityException se) throw se;
          if (cause instanceof Error err) throw err;
          missing.add(node); // ordinary/transient failure → exclude this node
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          missing.add(node);
        }
      }
    }
    if (!missing.isEmpty()) {
      throw new PartialResultException(
          op
              + " incomplete: "
              + missing.size()
              + " of "
              + allNodes.size()
              + " nodes unreachable within "
              + timeout,
          total,
          missing);
    }
    return total;
  }

  @Override
  public VectorCollectionConfig config() {
    return localCollection.config();
  }

  @Override
  public void close() {
    localCollection.close();
  }

  // ---- Builder ----

  /** Creates a new builder. */
  public static Builder builder() {
    return new Builder();
  }

  /** Fluent builder for {@link DistributedVectorCollection}. */
  public static final class Builder {

    private VectorCollection localCollection;
    private NodeId localNodeId;
    private NodeDirectory directory;
    private List<NodeId> allNodes;
    private Duration timeout = Duration.ofSeconds(5);
    private NodeCallContext nodeCallContext = NodeCallContext.none();

    private Builder() {}

    public Builder localCollection(VectorCollection localCollection) {
      this.localCollection = Objects.requireNonNull(localCollection);
      return this;
    }

    public Builder localNodeId(NodeId localNodeId) {
      this.localNodeId = Objects.requireNonNull(localNodeId);
      return this;
    }

    public Builder directory(NodeDirectory directory) {
      this.directory = Objects.requireNonNull(directory);
      return this;
    }

    public Builder allNodes(List<NodeId> allNodes) {
      this.allNodes = List.copyOf(Objects.requireNonNull(allNodes));
      return this;
    }

    public Builder allNodes(Set<NodeId> allNodes) {
      this.allNodes = List.copyOf(Objects.requireNonNull(allNodes));
      return this;
    }

    public Builder timeout(Duration timeout) {
      this.timeout = Objects.requireNonNull(timeout);
      return this;
    }

    public Builder nodeBearerToken(String token) {
      this.nodeCallContext = NodeCallContext.bearer(Objects.requireNonNull(token));
      return this;
    }

    public DistributedVectorCollection build() {
      Objects.requireNonNull(localCollection, "localCollection must be set");
      Objects.requireNonNull(localNodeId, "localNodeId must be set");
      Objects.requireNonNull(directory, "directory must be set");
      Objects.requireNonNull(allNodes, "allNodes must be set");
      return new DistributedVectorCollection(
          localCollection, localNodeId, directory, allNodes, timeout, nodeCallContext);
    }
  }
}
