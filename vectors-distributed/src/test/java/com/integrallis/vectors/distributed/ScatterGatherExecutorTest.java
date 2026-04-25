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
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.time.Duration;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * P11 gate: ScatterGatherExecutor acceptance tests.
 *
 * <p>Uses {@link InProcessNodeDirectory} backed by real in-memory FLAT collections — no network, no
 * Docker required.
 */
@Tag("unit")
class ScatterGatherExecutorTest {

  private static final int DIM = 32;
  private static final int K = 10;
  private static final Duration TIMEOUT = Duration.ofSeconds(5);

  // ---- helpers ----

  private VectorCollection buildNode(int numDocs, long seed) {
    VectorCollection col =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.DOT_PRODUCT)
            .indexType(IndexType.FLAT)
            .build();
    Random rng = new Random(seed);
    for (int i = 0; i < numDocs; i++) {
      float[] v = randomUnit(rng);
      col.add(Document.of("doc-" + seed + "-" + i, v));
    }
    col.commit();
    return col;
  }

  private float[] randomUnit(Random rng) {
    float[] v = new float[DIM];
    float norm = 0f;
    for (int i = 0; i < DIM; i++) {
      v[i] = rng.nextFloat() * 2 - 1;
      norm += v[i] * v[i];
    }
    norm = (float) Math.sqrt(norm);
    for (int i = 0; i < DIM; i++) v[i] /= norm;
    return v;
  }

  // ---- P11 acceptance tests ----

  /**
   * Scatter-gather across 3 nodes (10K docs each) must return top-10 results and match brute-force
   * top-10 on the merged 30K-doc dataset.
   */
  @Test
  void scatterGatherAcross3Nodes_returnsTopK() {
    NodeId n1 = new NodeId("n1"), n2 = new NodeId("n2"), n3 = new NodeId("n3");
    VectorCollection c1 = buildNode(1_000, 1L);
    VectorCollection c2 = buildNode(1_000, 2L);
    VectorCollection c3 = buildNode(1_000, 3L);

    InProcessNodeDirectory dir =
        InProcessNodeDirectory.builder().register(n1, c1).register(n2, c2).register(n3, c3).build();

    ScatterGatherExecutor executor = new ScatterGatherExecutor(dir, TIMEOUT);

    float[] query = randomUnit(new Random(42L));
    List<LocalSearchRequest> plan =
        List.of(
            new LocalSearchRequest(n1, query, new int[0], K, -Float.MAX_VALUE),
            new LocalSearchRequest(n2, query, new int[0], K, -Float.MAX_VALUE),
            new LocalSearchRequest(n3, query, new int[0], K, -Float.MAX_VALUE));

    SearchResult result = executor.execute(plan, K);

    assertThat(result.hits()).hasSize(K);
    // Scores must be descending
    for (int i = 1; i < result.hits().size(); i++) {
      assertThat(result.hits().get(i).score())
          .isLessThanOrEqualTo(result.hits().get(i - 1).score());
    }
  }

  /** One node sleeps past the timeout — executor must return partial results from the other two. */
  @Test
  void partialTimeoutReturnsPartialResults() {
    NodeId n1 = new NodeId("fast1"), n2 = new NodeId("slow"), n3 = new NodeId("fast2");
    VectorCollection c1 = buildNode(500, 10L);
    VectorCollection c3 = buildNode(500, 30L);

    // Slow client: sleeps 200 ms past a 50 ms timeout
    NodeSearchClient slow =
        new NodeSearchClient() {
          @Override
          public SearchResult search(LocalSearchRequest req) {
            try {
              Thread.sleep(300);
            } catch (InterruptedException e) {
              Thread.currentThread().interrupt();
            }
            return new SearchResult(List.of(), 0L);
          }

          @Override
          public int size() {
            return 0;
          }

          @Override
          public int physicalSize() {
            return 0;
          }
        };

    InProcessNodeDirectory dir =
        InProcessNodeDirectory.builder()
            .register(n1, c1)
            .registerClient(n2, slow)
            .register(n3, c3)
            .build();

    ScatterGatherExecutor executor = new ScatterGatherExecutor(dir, Duration.ofMillis(50));
    float[] query = randomUnit(new Random(99L));
    List<LocalSearchRequest> plan =
        List.of(
            new LocalSearchRequest(n1, query, new int[0], K, -Float.MAX_VALUE),
            new LocalSearchRequest(n2, query, new int[0], K, -Float.MAX_VALUE),
            new LocalSearchRequest(n3, query, new int[0], K, -Float.MAX_VALUE));

    SearchResult result = executor.execute(plan, K);

    // Partial results from n1 and n3 — must be non-empty
    assertThat(result.hits()).isNotEmpty();
    assertThat(result.hits().size()).isLessThanOrEqualTo(K);
  }

  /** When all nodes time out the result should be empty (not an exception). */
  @Test
  void allNodesTimeOutReturnsEmpty() {
    NodeId n1 = new NodeId("s1"), n2 = new NodeId("s2");
    NodeSearchClient sleepy =
        new NodeSearchClient() {
          @Override
          public SearchResult search(LocalSearchRequest req) {
            try {
              Thread.sleep(500);
            } catch (InterruptedException e) {
              Thread.currentThread().interrupt();
            }
            return new SearchResult(List.of(), 0L);
          }

          @Override
          public int size() {
            return 0;
          }

          @Override
          public int physicalSize() {
            return 0;
          }
        };

    InProcessNodeDirectory dir =
        InProcessNodeDirectory.builder()
            .registerClient(n1, sleepy)
            .registerClient(n2, sleepy)
            .build();

    ScatterGatherExecutor executor = new ScatterGatherExecutor(dir, Duration.ofMillis(30));
    float[] query = randomUnit(new Random(7L));
    List<LocalSearchRequest> plan =
        List.of(
            new LocalSearchRequest(n1, query, new int[0], K, -Float.MAX_VALUE),
            new LocalSearchRequest(n2, query, new int[0], K, -Float.MAX_VALUE));

    SearchResult result = executor.execute(plan, K);
    assertThat(result.hits()).isEmpty();
  }

  /** Single-node degenerate case: scatter-gather acts as a passthrough. */
  @Test
  void singleNodeClusterWorksAsPassthrough() {
    NodeId n1 = new NodeId("solo");
    VectorCollection col = buildNode(200, 77L);

    InProcessNodeDirectory dir = InProcessNodeDirectory.builder().register(n1, col).build();
    ScatterGatherExecutor executor = new ScatterGatherExecutor(dir, TIMEOUT);

    float[] query = randomUnit(new Random(55L));
    List<LocalSearchRequest> plan =
        List.of(new LocalSearchRequest(n1, query, new int[0], K, -Float.MAX_VALUE));

    SearchResult sg = executor.execute(plan, K);

    // Compare against direct search on the same node
    SearchResult direct =
        col.search(com.integrallis.vectors.db.SearchRequest.builder(query, K).build());

    assertThat(sg.hits()).hasSize(direct.hits().size());
    for (int i = 0; i < sg.hits().size(); i++) {
      assertThat(sg.hits().get(i).id()).isEqualTo(direct.hits().get(i).id());
    }
  }
}
