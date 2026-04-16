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
import java.util.List;
import java.util.Random;
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
}
