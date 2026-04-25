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

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Factory for creating in-JVM distributed test clusters.
 *
 * <p>Inspired by Hazelcast's {@code TestHazelcastInstanceFactory}. Creates N independent {@link
 * VectorCollection} instances wired together via {@link InProcessNodeDirectory}, enabling
 * scatter-gather testing without Docker.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * try (var factory = TestDistributedClusterFactory.builder()
 *     .nodeCount(3)
 *     .dimension(64)
 *     .metric(SimilarityFunction.COSINE)
 *     .indexType(IndexType.HNSW)
 *     .build()) {
 *
 *     DistributedVectorCollection cluster = factory.cluster();
 *     // All 3 nodes share mock network — test shard routing, TopK merge, gossip
 * }
 * }</pre>
 */
public final class TestDistributedClusterFactory implements AutoCloseable {

  private final List<VectorCollection> nodeCollections;
  private final List<NodeId> nodeIds;
  private final InProcessNodeDirectory directory;
  private final DistributedVectorCollection cluster;

  private TestDistributedClusterFactory(
      List<VectorCollection> nodeCollections,
      List<NodeId> nodeIds,
      InProcessNodeDirectory directory,
      DistributedVectorCollection cluster) {
    this.nodeCollections = nodeCollections;
    this.nodeIds = nodeIds;
    this.directory = directory;
    this.cluster = cluster;
  }

  /** Returns the distributed collection that fans out searches across all nodes. */
  public DistributedVectorCollection cluster() {
    return cluster;
  }

  /** Returns the underlying collection for node {@code index}. */
  public VectorCollection nodeCollection(int index) {
    return nodeCollections.get(index);
  }

  /** Returns the node id for node {@code index}. */
  public NodeId nodeId(int index) {
    return nodeIds.get(index);
  }

  /** Returns the number of nodes in the cluster. */
  public int nodeCount() {
    return nodeCollections.size();
  }

  /** Returns the node directory. */
  public InProcessNodeDirectory directory() {
    return directory;
  }

  /** Closes all underlying collections. */
  @Override
  public void close() {
    for (VectorCollection col : nodeCollections) {
      col.close();
    }
  }

  /** Creates a new builder. */
  public static Builder builder() {
    return new Builder();
  }

  /** Fluent builder for {@link TestDistributedClusterFactory}. */
  public static final class Builder {

    private int nodeCount = 3;
    private int dimension = 64;
    private SimilarityFunction metric = SimilarityFunction.COSINE;
    private IndexType indexType = IndexType.HNSW;
    private Duration timeout = Duration.ofSeconds(5);

    private Builder() {}

    /** Sets the number of cluster nodes. Default: 3. */
    public Builder nodeCount(int nodeCount) {
      if (nodeCount < 1) throw new IllegalArgumentException("nodeCount must be >= 1");
      this.nodeCount = nodeCount;
      return this;
    }

    /** Sets the vector dimension. Default: 64. */
    public Builder dimension(int dimension) {
      this.dimension = dimension;
      return this;
    }

    /** Sets the similarity function. Default: COSINE. */
    public Builder metric(SimilarityFunction metric) {
      this.metric = Objects.requireNonNull(metric);
      return this;
    }

    /** Sets the index type. Default: HNSW. */
    public Builder indexType(IndexType indexType) {
      this.indexType = Objects.requireNonNull(indexType);
      return this;
    }

    /** Sets the scatter-gather timeout. Default: 5 seconds. */
    public Builder timeout(Duration timeout) {
      this.timeout = Objects.requireNonNull(timeout);
      return this;
    }

    /** Builds the cluster factory with N independent node collections. */
    public TestDistributedClusterFactory build() {
      List<NodeId> nodeIds = new ArrayList<>(nodeCount);
      List<VectorCollection> collections = new ArrayList<>(nodeCount);
      InProcessNodeDirectory.Builder dirBuilder = InProcessNodeDirectory.builder();

      for (int i = 0; i < nodeCount; i++) {
        NodeId id = new NodeId("node-" + i);
        nodeIds.add(id);
        VectorCollection col =
            VectorCollection.builder()
                .dimension(dimension)
                .metric(metric)
                .indexType(indexType)
                .build();
        collections.add(col);
        dirBuilder.register(id, col);
      }

      InProcessNodeDirectory directory = dirBuilder.build();

      DistributedVectorCollection cluster =
          DistributedVectorCollection.builder()
              .localCollection(collections.get(0))
              .localNodeId(nodeIds.get(0))
              .directory(directory)
              .allNodes(nodeIds)
              .timeout(timeout)
              .build();

      return new TestDistributedClusterFactory(collections, nodeIds, directory, cluster);
    }
  }
}
