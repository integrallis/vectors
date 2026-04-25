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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class TestDistributedClusterFactoryTest {

  private static final int DIM = 32;
  private static final long SEED = 42L;

  @Nested
  @Tag("unit")
  class ClusterCreation {

    @Test
    void createsClusterWithCorrectNodeCount() {
      try (var factory =
          TestDistributedClusterFactory.builder()
              .nodeCount(3)
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .build()) {
        assertThat(factory.nodeCount()).isEqualTo(3);
        assertThat(factory.cluster()).isNotNull();
        assertThat(factory.directory().allNodes()).hasSize(3);
      }
    }

    @Test
    void singleNodeClusterWorks() {
      try (var factory =
          TestDistributedClusterFactory.builder()
              .nodeCount(1)
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .build()) {
        assertThat(factory.nodeCount()).isEqualTo(1);
      }
    }

    @Test
    void fiveNodeClusterWorks() {
      try (var factory =
          TestDistributedClusterFactory.builder()
              .nodeCount(5)
              .dimension(DIM)
              .metric(SimilarityFunction.COSINE)
              .indexType(IndexType.VAMANA)
              .build()) {
        assertThat(factory.nodeCount()).isEqualTo(5);
      }
    }
  }

  @Nested
  @Tag("unit")
  class ScatterGatherSearch {

    @Test
    void searchAcrossNodesReturnsResults() {
      try (var factory =
          TestDistributedClusterFactory.builder()
              .nodeCount(3)
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .build()) {

        Random rng = new Random(SEED);
        // Distribute docs across 3 nodes (100 each)
        for (int n = 0; n < 3; n++) {
          List<Document> docs = new ArrayList<>(100);
          for (int i = 0; i < 100; i++) {
            float[] v = new float[DIM];
            for (int d = 0; d < DIM; d++) v[d] = rng.nextFloat() - 0.5f;
            docs.add(new Document("node" + n + "-doc-" + i, v, null, null));
          }
          factory.nodeCollection(n).addAll(docs);
          factory.nodeCollection(n).commit();
        }

        // Verify aggregate size
        assertThat(factory.cluster().size()).isEqualTo(300);

        // Search the cluster
        float[] query = new float[DIM];
        for (int d = 0; d < DIM; d++) query[d] = rng.nextFloat() - 0.5f;

        SearchResult result = factory.cluster().search(SearchRequest.builder(query, 10).build());
        assertThat(result.hits()).hasSize(10);

        // Verify results come from multiple nodes
        long distinctNodePrefixes =
            result.hits().stream().map(h -> h.id().substring(0, 5)).distinct().count();
        assertThat(distinctNodePrefixes)
            .as("results should come from multiple nodes")
            .isGreaterThanOrEqualTo(2);
      }
    }
  }
}
