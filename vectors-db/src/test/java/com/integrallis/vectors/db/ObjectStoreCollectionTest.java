/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.filter.Filters;
import com.integrallis.vectors.storage.backend.LocalFileStorageBackend;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Verifies {@link VectorCollectionBuilder#objectStore} — a full-featured collection backed by an
 * object-storage durable floor. A producer commits to the store; a second collection with a fresh,
 * empty local cache hydrates from the store on {@code build()} and serves the data, with metadata
 * filters and text preserved through the round trip. A {@link LocalFileStorageBackend} stands in
 * for S3 / R2 / MinIO (all are the same {@link
 * com.integrallis.vectors.storage.backend.StorageBackend} SPI).
 */
@Tag("unit")
class ObjectStoreCollectionTest {

  private static final int DIM = 8;

  private static float[] vec(float base) {
    float[] v = new float[DIM];
    for (int i = 0; i < DIM; i++) {
      v[i] = base + i * 0.01f;
    }
    return v;
  }

  @Test
  void freshCacheHydratesFromObjectStoreWithFiltersAndText(@TempDir Path tmp) throws Exception {
    LocalFileStorageBackend objectStore = new LocalFileStorageBackend(tmp.resolve("bucket"));
    String prefix = "collections/docs/";

    // Producer: a normal HNSW collection whose durable floor is the object store.
    try (VectorCollection producer =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.HNSW)
            .storagePath(tmp.resolve("cache-a"))
            .objectStore(objectStore, prefix)
            .build()) {
      producer.add(new Document("a", vec(1f), "alpha", Map.of("tag", MetadataValue.of("x"))));
      producer.add(new Document("b", vec(2f), "beta", Map.of("tag", MetadataValue.of("y"))));
      producer.add(new Document("c", vec(3f), "gamma", Map.of("tag", MetadataValue.of("x"))));
      producer.commit();
      // Shipping is asynchronous; wait for the durable CURRENT pointer to land.
      waitFor(() -> objectStore.get(prefix + "CURRENT") != null, "shipped CURRENT");
    }

    // Consumer: a brand-new collection with an EMPTY local cache, same store + prefix.
    // build() hydrates the local cache from object storage before opening.
    try (VectorCollection consumer =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.HNSW)
            .storagePath(tmp.resolve("cache-b"))
            .objectStore(objectStore, prefix)
            .build()) {

      assertThat(consumer.size()).isEqualTo(3);

      // Nearest-neighbour search works off the hydrated cache.
      var top = consumer.search(SearchRequest.builder(vec(1f), 1).build()).hits();
      assertThat(top.get(0).id()).isEqualTo("a");

      // Metadata filters survive the object-storage round trip.
      var filtered =
          consumer
              .search(SearchRequest.builder(vec(1f), 10).filter(Filters.eq("tag", "x")).build())
              .hits();
      assertThat(filtered).extracting(SearchResult.Hit::id).containsExactlyInAnyOrder("a", "c");

      // Text payloads survive too.
      assertThat(consumer.get("b").text()).isEqualTo("beta");
    }
  }

  private interface Condition {
    boolean met() throws Exception;
  }

  private static void waitFor(Condition condition, String what) throws Exception {
    long deadline = System.nanoTime() + 10_000_000_000L; // 10s
    while (System.nanoTime() < deadline) {
      if (condition.met()) {
        return;
      }
      Thread.sleep(20);
    }
    throw new AssertionError("timed out waiting for " + what);
  }
}
