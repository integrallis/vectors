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
package com.integrallis.vectors.cluster;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.filter.Filters;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class ClusterVectorCollectionTest {

  private static final int DIM = 32;
  private static final int SHARDS = 4;

  private static VectorCollection flatShard(Path shardPath) {
    var builder =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.DOT_PRODUCT)
            .indexType(IndexType.FLAT);
    if (shardPath != null) {
      builder.storagePath(shardPath);
    }
    return builder.build();
  }

  private static ClusterVectorCollection inMemoryCluster() {
    return ClusterVectorCollection.builder()
        .shardCount(SHARDS)
        .shardFactory((idx, path) -> flatShard(path))
        .build();
  }

  private static float[] randomUnit(Random rng) {
    float[] v = new float[DIM];
    float norm = 0f;
    for (int i = 0; i < DIM; i++) {
      v[i] = rng.nextFloat() * 2 - 1;
      norm += v[i] * v[i];
    }
    norm = (float) Math.sqrt(norm);
    for (int i = 0; i < DIM; i++) {
      v[i] /= norm;
    }
    return v;
  }

  @Test
  void documentsArePartitionedAcrossShardsAndRetrievable() {
    Random rng = new Random(1L);
    List<VectorCollection> shards = new ArrayList<>();
    for (int i = 0; i < SHARDS; i++) {
      shards.add(flatShard(null));
    }
    ClusterVectorCollection cluster = ClusterVectorCollection.over(shards);

    int n = 400;
    for (int i = 0; i < n; i++) {
      cluster.add(Document.of("doc-" + i, randomUnit(rng)));
    }
    cluster.commit();

    // The per-shard sizes must sum to the total, and the partition must be non-trivial.
    int summed = 0;
    int nonEmptyShards = 0;
    for (VectorCollection shard : shards) {
      summed += shard.size();
      if (shard.size() > 0) {
        nonEmptyShards++;
      }
    }
    assertThat(summed).isEqualTo(n);
    assertThat(cluster.size()).isEqualTo(n);
    assertThat(nonEmptyShards).isEqualTo(SHARDS);

    // Every document is retrievable through the cluster and lands on the shard the router chose.
    for (int i = 0; i < n; i++) {
      String id = "doc-" + i;
      assertThat(cluster.contains(id)).isTrue();
      assertThat(cluster.get(id)).isNotNull();
      int owner = cluster.router().route(id);
      assertThat(shards.get(owner).contains(id)).isTrue();
    }
    cluster.close();
  }

  @Test
  void searchMatchesSingleCollectionGroundTruth() {
    Random rng = new Random(42L);
    int n = 600;
    int k = 10;

    List<Document> corpus = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      corpus.add(Document.of("v-" + i, randomUnit(rng)));
    }

    ClusterVectorCollection cluster = inMemoryCluster();
    cluster.addAll(corpus);
    cluster.commit();

    VectorCollection reference = flatShard(null);
    reference.addAll(corpus);
    reference.commit();

    for (int q = 0; q < 25; q++) {
      float[] query = randomUnit(rng);
      SearchRequest req = SearchRequest.builder(query, k).build();
      SearchResult clusterResult = cluster.search(req);
      SearchResult referenceResult = reference.search(req);

      assertThat(clusterResult.hits()).hasSameSizeAs(referenceResult.hits());
      // FLAT is exact; the scatter-gather + top-k merge must reproduce the global ranking exactly.
      // Scores are compared within a tight epsilon: the same query·doc dot product can differ by a
      // ULP between a full index and a smaller per-shard index (SIMD reduction/layout), but the
      // resulting ranking is identical.
      for (int i = 0; i < referenceResult.hits().size(); i++) {
        assertThat(clusterResult.hits().get(i).id()).isEqualTo(referenceResult.hits().get(i).id());
        assertThat(clusterResult.hits().get(i).score())
            .isCloseTo(referenceResult.hits().get(i).score(), within(1e-4f));
      }
    }
    cluster.close();
    reference.close();
  }

  @Test
  void filteredSearchSpansAllShards() {
    Random rng = new Random(7L);
    ClusterVectorCollection cluster = inMemoryCluster();

    int reds = 0;
    for (int i = 0; i < 400; i++) {
      String color = (i % 2 == 0) ? "red" : "blue";
      if (color.equals("red")) {
        reds++;
      }
      Document doc =
          new Document("doc-" + i, randomUnit(rng), null, Map.of("color", MetadataValue.of(color)));
      cluster.add(doc);
    }
    cluster.commit();

    // Request more hits than there are reds; every returned hit must satisfy the filter, and the
    // result must draw from whichever shards happen to hold red documents.
    SearchRequest req =
        SearchRequest.builder(randomUnit(rng), reds + 50)
            .filter(Filters.eq("color", "red"))
            .build();
    SearchResult result = cluster.search(req);

    assertThat(result.hits()).isNotEmpty();
    assertThat(result.hits()).hasSizeLessThanOrEqualTo(reds);
    for (SearchResult.Hit hit : result.hits()) {
      MetadataValue color = hit.document().metadata().get("color");
      assertThat(((MetadataValue.Str) color).value()).isEqualTo("red");
    }
    cluster.close();
  }

  @Test
  void deleteRoutesToOwningShardAndRemovesFromSearch() {
    Random rng = new Random(2L);
    ClusterVectorCollection cluster = inMemoryCluster();

    float[] target = randomUnit(rng);
    cluster.add(Document.of("target", target));
    for (int i = 0; i < 100; i++) {
      cluster.add(Document.of("other-" + i, randomUnit(rng)));
    }
    cluster.commit();

    assertThat(cluster.search(SearchRequest.builder(target, 1).build()).hits().get(0).id())
        .isEqualTo("target");

    assertThat(cluster.delete("target")).isTrue();
    cluster.commit();

    SearchResult after = cluster.search(SearchRequest.builder(target, 5).build());
    assertThat(after.hits()).extracting(SearchResult.Hit::id).doesNotContain("target");
    assertThat(cluster.contains("target")).isFalse();
    cluster.close();
  }

  @Test
  void deleteWhereFansOutAcrossShards() {
    Random rng = new Random(3L);
    ClusterVectorCollection cluster = inMemoryCluster();
    for (int i = 0; i < 300; i++) {
      String color = (i % 3 == 0) ? "drop" : "keep";
      cluster.add(
          new Document(
              "doc-" + i, randomUnit(rng), null, Map.of("color", MetadataValue.of(color))));
    }
    cluster.commit();

    int before = cluster.size();
    int deleted = cluster.deleteWhere(Filters.eq("color", "drop"));
    cluster.commit();

    assertThat(deleted).isEqualTo(100);
    assertThat(cluster.size()).isEqualTo(before - 100);
    cluster.close();
  }

  @Test
  void upsertReplacesOnOwningShard() {
    ClusterVectorCollection cluster = inMemoryCluster();
    float[] first = randomUnit(new Random(10L));
    float[] second = randomUnit(new Random(11L));

    cluster.add(Document.of("dup", first, "first"));
    cluster.commit();
    cluster.upsert(Document.of("dup", second, "second"));
    cluster.commit();

    assertThat(cluster.size()).isEqualTo(1);
    assertThat(cluster.get("dup").text()).isEqualTo("second");
    cluster.close();
  }

  @Test
  void singleShardClusterPassesThroughToUnderlyingSearch() {
    Random rng = new Random(5L);
    VectorCollection solo = flatShard(null);
    ClusterVectorCollection cluster = ClusterVectorCollection.over(List.of(solo));
    for (int i = 0; i < 200; i++) {
      cluster.add(Document.of("s-" + i, randomUnit(rng)));
    }
    cluster.commit();

    float[] query = randomUnit(new Random(77L));
    SearchResult viaCluster = cluster.search(SearchRequest.builder(query, 10).build());
    SearchResult direct = solo.search(SearchRequest.builder(query, 10).build());

    assertThat(viaCluster.hits()).hasSameSizeAs(direct.hits());
    for (int i = 0; i < direct.hits().size(); i++) {
      assertThat(viaCluster.hits().get(i).id()).isEqualTo(direct.hits().get(i).id());
    }
    cluster.close();
  }

  @Test
  void shardsAreIndependentlyDurableAcrossReopen(@TempDir Path root) {
    Random rng = new Random(99L);
    List<Document> corpus = new ArrayList<>();
    for (int i = 0; i < 300; i++) {
      corpus.add(Document.of("d-" + i, randomUnit(rng)));
    }

    ClusterVectorCollection cluster =
        ClusterVectorCollection.builder()
            .shardCount(3)
            .storageRoot(root)
            .shardFactory((idx, path) -> flatShard(path))
            .build();
    cluster.addAll(corpus);
    cluster.commit();
    assertThat(cluster.size()).isEqualTo(300);
    cluster.close();

    // Reopen each shard directory independently and wrap with the same shard order.
    List<VectorCollection> reopened = new ArrayList<>();
    for (int i = 0; i < 3; i++) {
      reopened.add(flatShard(root.resolve("shard-" + i)));
    }
    ClusterVectorCollection reloaded = ClusterVectorCollection.over(reopened);

    assertThat(reloaded.size()).isEqualTo(300);
    // A known document is found, and routing still resolves it to the shard it was persisted on.
    Document probe = corpus.get(123);
    SearchResult res = reloaded.search(SearchRequest.builder(probe.vector(), 1).build());
    assertThat(res.hits().get(0).id()).isEqualTo("d-123");
    assertThat(reloaded.get("d-123")).isNotNull();
    reloaded.close();
  }

  @Test
  void rejectsHeterogeneousShards() {
    VectorCollection dim32 = flatShard(null);
    VectorCollection dim16 =
        VectorCollection.builder()
            .dimension(16)
            .metric(SimilarityFunction.DOT_PRODUCT)
            .indexType(IndexType.FLAT)
            .build();
    assertThatThrownBy(() -> ClusterVectorCollection.over(List.of(dim32, dim16)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dimension");
    dim32.close();
    dim16.close();
  }
}
