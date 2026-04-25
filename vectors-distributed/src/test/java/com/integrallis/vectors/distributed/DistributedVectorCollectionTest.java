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

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.Document;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * P12 gate: DistributedVectorCollection acceptance tests.
 *
 * <p>3-node cluster, in-process, no network. Each test rebuilds all collections fresh.
 */
@Tag("unit")
class DistributedVectorCollectionTest {

  private static final int DIM = 32;
  private static final int K = 10;
  private static final Duration TIMEOUT = Duration.ofSeconds(5);

  private NodeId local, peer1, peer2;
  private VectorCollection localCol, peerCol1, peerCol2;
  private InProcessNodeDirectory directory;
  private DistributedVectorCollection dvc;

  @BeforeEach
  void setUp() {
    local = new NodeId("local");
    peer1 = new NodeId("peer1");
    peer2 = new NodeId("peer2");

    localCol = emptyCollection();
    peerCol1 = emptyCollection();
    peerCol2 = emptyCollection();

    directory =
        InProcessNodeDirectory.builder()
            .register(local, localCol)
            .register(peer1, peerCol1)
            .register(peer2, peerCol2)
            .build();

    dvc =
        DistributedVectorCollection.builder()
            .localCollection(localCol)
            .localNodeId(local)
            .directory(directory)
            .allNodes(List.of(local, peer1, peer2))
            .timeout(TIMEOUT)
            .build();
  }

  private VectorCollection emptyCollection() {
    return VectorCollection.builder()
        .dimension(DIM)
        .metric(SimilarityFunction.DOT_PRODUCT)
        .indexType(IndexType.FLAT)
        .build();
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

  /** Adding 100 docs to the distributed collection, committing, and searching must find them. */
  @Test
  void addSearchConsistency_3nodes() {
    Random rng = new Random(1L);
    List<float[]> vectors = new ArrayList<>();
    for (int i = 0; i < 100; i++) {
      float[] v = randomUnit(rng);
      vectors.add(v);
      dvc.add(Document.of("doc-" + i, v));
    }
    dvc.commit();

    // Size should include the 100 docs on the local node (peers are empty)
    assertThat(dvc.size()).isEqualTo(100);

    // Search with the first doc's vector — it must be in the top-1 result
    SearchResult result = dvc.search(SearchRequest.builder(vectors.get(0), 1).build());
    assertThat(result.hits()).hasSize(1);
    assertThat(result.hits().get(0).id()).isEqualTo("doc-0");
  }

  /** Deleting a document and committing must exclude it from subsequent searches. */
  @Test
  void deletePropagatesToSearchResults() {
    Random rng = new Random(2L);
    float[] targetVec = randomUnit(rng);
    dvc.add(Document.of("target", targetVec));
    for (int i = 0; i < 20; i++) {
      dvc.add(Document.of("other-" + i, randomUnit(rng)));
    }
    dvc.commit();

    // Verify target is found before deletion
    SearchResult before = dvc.search(SearchRequest.builder(targetVec, 1).build());
    assertThat(before.hits().get(0).id()).isEqualTo("target");

    // Delete and commit
    boolean deleted = dvc.delete("target");
    assertThat(deleted).isTrue();
    dvc.commit();

    // Target must not appear in top-5 results anymore
    SearchResult after = dvc.search(SearchRequest.builder(targetVec, 5).build());
    assertThat(after.hits()).extracting(h -> h.id()).doesNotContain("target");
  }

  /** When one peer node throws, results from the other two must still be returned. */
  @Test
  void nodeFailureReturnsPartialResults() {
    // Pre-load local and one peer
    Random rng = new Random(3L);
    for (int i = 0; i < 50; i++) {
      peerCol1.add(Document.of("p1-" + i, randomUnit(rng)));
    }
    peerCol1.commit();
    for (int i = 0; i < 50; i++) {
      localCol.add(Document.of("local-" + i, randomUnit(rng)));
    }
    localCol.commit();

    // Replace peer2 with a client that always throws
    NodeSearchClient broken =
        new NodeSearchClient() {
          @Override
          public SearchResult search(LocalSearchRequest req) {
            throw new RuntimeException("simulated node failure");
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

    InProcessNodeDirectory dirWithFailure =
        InProcessNodeDirectory.builder()
            .register(local, localCol)
            .register(peer1, peerCol1)
            .registerClient(peer2, broken)
            .build();

    DistributedVectorCollection dvcWithFailure =
        DistributedVectorCollection.builder()
            .localCollection(localCol)
            .localNodeId(local)
            .directory(dirWithFailure)
            .allNodes(List.of(local, peer1, peer2))
            .timeout(TIMEOUT)
            .build();

    float[] query = randomUnit(new Random(99L));
    SearchResult result = dvcWithFailure.search(SearchRequest.builder(query, K).build());

    // Partial: results from local + peer1 (peer2 failed)
    assertThat(result.hits()).isNotEmpty();
    assertThat(result.hits().size()).isLessThanOrEqualTo(K);
  }

  /**
   * 3 FLAT nodes × 3K docs each (9K total). Brute-force ground truth computed on the combined
   * corpus; recall@10 from scatter-gather must be ≥ 0.80 for a random query.
   *
   * <p>Deliberately smaller than the design's 1M-doc case to keep CI fast — the slow {@link
   * org.junit.jupiter.api.Tag} prevents this from running in the default test task.
   */
  @Test
  @Tag("slow")
  void scatterGatherRecall_3nodes() {
    int n = 3_000; // docs per node
    int totalNodes = 3;
    int kRecall = 10;
    Random rng = new Random(999L);

    // Build corpus across 3 separate collections
    List<VectorCollection> nodeCollections = new ArrayList<>();
    List<float[]> allVectors = new ArrayList<>();
    List<String> allIds = new ArrayList<>();

    for (int nodeIdx = 0; nodeIdx < totalNodes; nodeIdx++) {
      VectorCollection col = emptyCollection();
      for (int i = 0; i < n; i++) {
        float[] v = randomUnit(rng);
        String id = "n" + nodeIdx + "-" + i;
        col.add(Document.of(id, v));
        allVectors.add(v);
        allIds.add(id);
      }
      col.commit();
      nodeCollections.add(col);
    }

    // Build brute-force reference collection on the merged corpus
    VectorCollection bruteForce = emptyCollection();
    for (int i = 0; i < allVectors.size(); i++) {
      bruteForce.add(Document.of(allIds.get(i), allVectors.get(i)));
    }
    bruteForce.commit();

    // Wire 3 nodes into a distributed collection
    InProcessNodeDirectory.Builder dirBuilder = InProcessNodeDirectory.builder();
    List<NodeId> nodeIds = new ArrayList<>();
    for (int nodeIdx = 0; nodeIdx < totalNodes; nodeIdx++) {
      NodeId nid = new NodeId("recall-n" + nodeIdx);
      nodeIds.add(nid);
      dirBuilder.register(nid, nodeCollections.get(nodeIdx));
    }
    InProcessNodeDirectory dir = dirBuilder.build();

    DistributedVectorCollection dvCol =
        DistributedVectorCollection.builder()
            .localCollection(nodeCollections.get(0))
            .localNodeId(nodeIds.get(0))
            .directory(dir)
            .allNodes(nodeIds)
            .timeout(TIMEOUT)
            .build();

    // Run 20 random queries and measure recall@10
    int hitCount = 0;
    int queryCount = 20;
    for (int q = 0; q < queryCount; q++) {
      float[] query = randomUnit(rng);
      SearchResult sgResult = dvCol.search(SearchRequest.builder(query, kRecall).build());
      SearchResult bfResult = bruteForce.search(SearchRequest.builder(query, kRecall).build());

      Set<String> bfIds = new HashSet<>();
      for (SearchResult.Hit h : bfResult.hits()) bfIds.add(h.id());
      for (SearchResult.Hit h : sgResult.hits()) {
        if (bfIds.contains(h.id())) hitCount++;
      }
    }

    double recall = (double) hitCount / (queryCount * kRecall);
    assertThat(recall).as("recall@10 across 3 nodes").isGreaterThanOrEqualTo(0.80);

    bruteForce.close();
    nodeCollections.forEach(VectorCollection::close);
  }

  /** Degenerate 1-node cluster behaves identically to a plain VectorCollection search. */
  @Test
  void singleNodePassthrough() {
    Random rng = new Random(5L);
    NodeId soloNode = new NodeId("solo");
    VectorCollection soloCol = emptyCollection();
    for (int i = 0; i < 200; i++) {
      soloCol.add(Document.of("s-" + i, randomUnit(rng)));
    }
    soloCol.commit();

    InProcessNodeDirectory soloDir =
        InProcessNodeDirectory.builder().register(soloNode, soloCol).build();
    DistributedVectorCollection soloDvc =
        DistributedVectorCollection.builder()
            .localCollection(soloCol)
            .localNodeId(soloNode)
            .directory(soloDir)
            .allNodes(List.of(soloNode))
            .timeout(TIMEOUT)
            .build();

    float[] query = randomUnit(new Random(77L));
    SearchResult dvResult = soloDvc.search(SearchRequest.builder(query, K).build());
    SearchResult direct = soloCol.search(SearchRequest.builder(query, K).build());

    assertThat(dvResult.hits()).hasSameSizeAs(direct.hits());
    for (int i = 0; i < dvResult.hits().size(); i++) {
      assertThat(dvResult.hits().get(i).id()).isEqualTo(direct.hits().get(i).id());
    }
  }

  /**
   * BuoyIndex gossip convergence gate (design section 9.4).
   *
   * <p>Three nodes share a {@link GossipClusterMembership}. Each registers a listener. One node
   * announces a new BuoyIndex version hash; all three listeners must receive a {@link
   * ClusterMembership.MembershipEvent.BuoyIndexUpdated} event with the correct hash within 500 ms.
   *
   * <p>In this in-process simulation, propagation is synchronous (the call to {@code
   * announceVersion} fans out inline), so convergence is instantaneous. The 500 ms deadline is
   * enforced via {@link CountDownLatch} to demonstrate the contract and remain valid when replaced
   * by an async implementation.
   */
  @Test
  void buoyIndexGossipConverges() throws InterruptedException {
    NodeId n1 = new NodeId("gossip-n1");
    NodeId n2 = new NodeId("gossip-n2");
    NodeId n3 = new NodeId("gossip-n3");

    GossipClusterMembership gossip = new GossipClusterMembership(Set.of(n1, n2, n3));

    // Track events received by each of the three "nodes"
    List<String> receivedHashes = new CopyOnWriteArrayList<>();
    int totalListeners = 3;
    CountDownLatch latch = new CountDownLatch(totalListeners);

    for (int i = 0; i < totalListeners; i++) {
      gossip.registerChangeListener(
          event -> {
            if (event instanceof ClusterMembership.MembershipEvent.BuoyIndexUpdated bu) {
              receivedHashes.add(bu.newVersionHash());
              latch.countDown();
            }
          });
    }

    // One node announces a new BuoyIndex version (simulates completing a k-means training pass)
    String newHash = "sha256-buoy-v2-abc123";
    gossip.announceVersion(newHash);

    // All 3 listeners must have received the event within 500 ms
    boolean converged = latch.await(500, TimeUnit.MILLISECONDS);
    assertThat(converged).as("All nodes must receive BuoyIndexUpdated within 500 ms").isTrue();
    assertThat(receivedHashes).hasSize(totalListeners).containsOnly(newHash);
    assertThat(gossip.currentVersionHash()).isEqualTo(newHash);
  }

  /**
   * Announcing the same BuoyIndex version twice must not produce duplicate events (idempotency).
   */
  @Test
  void buoyIndexGossip_idempotentAnnouncement() {
    NodeId n1 = new NodeId("idem-n1");
    NodeId n2 = new NodeId("idem-n2");
    GossipClusterMembership gossip = new GossipClusterMembership(Set.of(n1, n2));

    List<String> receivedHashes = new CopyOnWriteArrayList<>();
    gossip.registerChangeListener(
        event -> {
          if (event instanceof ClusterMembership.MembershipEvent.BuoyIndexUpdated bu) {
            receivedHashes.add(bu.newVersionHash());
          }
        });

    gossip.announceVersion("v1");
    gossip.announceVersion("v1"); // duplicate — must not fire a second event
    gossip.announceVersion("v2"); // new version — must fire

    assertThat(receivedHashes).containsExactly("v1", "v2");
  }

  /** Node join and leave events are propagated to all registered listeners. */
  @Test
  void gossipMembership_joinAndLeaveEvents() {
    NodeId n1 = new NodeId("jl-n1");
    NodeId n2 = new NodeId("jl-n2");
    GossipClusterMembership gossip = new GossipClusterMembership(Set.of(n1, n2));

    List<ClusterMembership.MembershipEvent> received = new CopyOnWriteArrayList<>();
    gossip.registerChangeListener(received::add);

    NodeId n3 = new NodeId("jl-n3");
    gossip.join(n3);
    assertThat(gossip.liveNodes()).contains(n1, n2, n3);

    gossip.leave(n2);
    assertThat(gossip.liveNodes()).contains(n1, n3).doesNotContain(n2);

    assertThat(received).hasSize(2);
    assertThat(received.get(0)).isInstanceOf(ClusterMembership.MembershipEvent.NodeJoined.class);
    assertThat(received.get(1)).isInstanceOf(ClusterMembership.MembershipEvent.NodeLeft.class);
  }
}
