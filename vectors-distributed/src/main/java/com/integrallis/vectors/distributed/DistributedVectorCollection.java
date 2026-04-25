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

import com.integrallis.vectors.db.Document;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.VectorCollectionConfig;
import com.integrallis.vectors.db.filter.Filter;
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

  private DistributedVectorCollection(
      VectorCollection localCollection,
      NodeId localNodeId,
      NodeDirectory directory,
      List<NodeId> allNodes,
      Duration timeout) {
    this.localCollection = localCollection;
    this.localNodeId = localNodeId;
    this.directory = directory;
    this.allNodes = List.copyOf(allNodes);
    this.executor = new ScatterGatherExecutor(directory, timeout);
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
    // Build a broadcast plan: one LocalSearchRequest per node, empty clusterIds = search all
    List<LocalSearchRequest> plan = new ArrayList<>(allNodes.size());
    for (NodeId node : allNodes) {
      plan.add(
          new LocalSearchRequest(
              node, request.query(), new int[0], request.k(), request.minScore()));
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

  /** Aggregates sizes across all nodes in the directory. */
  @Override
  public int size() {
    int total = 0;
    for (NodeId node : allNodes) {
      total += directory.clientFor(node).size();
    }
    return total;
  }

  /** Aggregates physical sizes across all nodes. */
  @Override
  public int physicalSize() {
    int total = 0;
    for (NodeId node : allNodes) {
      total += directory.clientFor(node).physicalSize();
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

    public DistributedVectorCollection build() {
      Objects.requireNonNull(localCollection, "localCollection must be set");
      Objects.requireNonNull(localNodeId, "localNodeId must be set");
      Objects.requireNonNull(directory, "directory must be set");
      Objects.requireNonNull(allNodes, "allNodes must be set");
      return new DistributedVectorCollection(
          localCollection, localNodeId, directory, allNodes, timeout);
    }
  }
}
