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
package com.integrallis.vectors.ingest.sinks;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.ingest.Batch;
import com.integrallis.vectors.ingest.EmbeddedDoc;
import com.integrallis.vectors.ingest.IngestDoc;
import com.integrallis.vectors.ivf.ClusterSplitter;
import com.integrallis.vectors.ivf.DistributedVectorCollection;
import com.integrallis.vectors.ivf.IvfBuildParams;
import com.integrallis.vectors.ivf.IvfHit;
import com.integrallis.vectors.ivf.TierPolicy;
import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class DistributedVectorSinkTest {

  private static final int DIM = 8;
  private static final SimilarityFunction METRIC = SimilarityFunction.COSINE;

  private static IvfBuildParams params() {
    return new IvfBuildParams(2, 30, 0f, false, 42L, 0);
  }

  private static ClusterSplitter splitter() {
    return new ClusterSplitter(10_000, 30, 42L);
  }

  private static TierPolicy policy() {
    return new TierPolicy(5, 2);
  }

  private static Batch batch(long batchId, int n, long startOffset, long seed) {
    Random rng = new Random(seed);
    List<EmbeddedDoc> docs = new ArrayList<>(n);
    for (int i = 0; i < n; i++) {
      float[] v = new float[DIM];
      for (int d = 0; d < DIM; d++) v[d] = rng.nextFloat() * 2f - 1f;
      docs.add(new EmbeddedDoc(IngestDoc.text("id-" + (startOffset + i), "x"), v, startOffset + i));
    }
    return new Batch(batchId, docs);
  }

  @Test
  void bootstrappingFlowBuildsOnFirstCommit(@TempDir Path tmp) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    DistributedVectorSink sink =
        DistributedVectorSink.bootstrapping(tmp, t3, params(), splitter(), policy(), METRIC);
    assertThat(sink.collection()).isNull();

    sink.addAll(batch(0, 16, 0, 1L));
    assertThat(sink.collection()).isNull(); // build deferred to commit
    sink.commit();
    assertThat(sink.collection()).isNotNull();
    assertThat(sink.committedCount()).isEqualTo(16L);

    sink.addAll(batch(1, 8, 16, 2L));
    sink.commit();
    assertThat(sink.committedCount()).isEqualTo(24L);

    DistributedVectorCollection col = sink.collection();
    assertThat(col.size()).isEqualTo(24);
    sink.close();
  }

  @Test
  void bootstrappingRequiresFirstBatchToCommit(@TempDir Path tmp) {
    HeapStorageBackend t3 = new HeapStorageBackend();
    DistributedVectorSink sink =
        DistributedVectorSink.bootstrapping(tmp, t3, params(), splitter(), policy(), METRIC);
    assertThatThrownBy(sink::commit)
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("non-empty first batch");
  }

  @Test
  void openingFlowReusesExistingCollection(@TempDir Path tmp) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    // First, build a collection with the bootstrapping sink and close it.
    try (DistributedVectorSink first =
        DistributedVectorSink.bootstrapping(tmp, t3, params(), splitter(), policy(), METRIC)) {
      first.addAll(batch(0, 12, 0, 7L));
      first.commit();
    }
    // Reopen via opening factory.
    try (DistributedVectorSink reopened =
        DistributedVectorSink.opening(tmp, t3, METRIC, policy())) {
      assertThat(reopened.collection().size()).isEqualTo(12);
      reopened.addAll(batch(1, 4, 12, 8L));
      reopened.commit();
      assertThat(reopened.committedCount()).isEqualTo(16L);
    }
  }

  @Test
  void searchableAfterIngest(@TempDir Path tmp) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    try (DistributedVectorSink sink =
        DistributedVectorSink.bootstrapping(tmp, t3, params(), splitter(), policy(), METRIC)) {
      Batch b = batch(0, 20, 0, 11L);
      sink.addAll(b);
      sink.commit();
      DistributedVectorCollection col = sink.collection();
      // Search using the first vector and confirm we hit its own id.
      float[] q = b.docs().get(0).vector();
      List<IvfHit> hits = col.search(q, 1, 2);
      assertThat(hits).isNotEmpty();
      assertThat(hits.get(0).id()).isEqualTo(b.docs().get(0).doc().id());
    }
  }

  @Test
  void walRotationFiresOnCadence(@TempDir Path tmp) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    try (DistributedVectorSink sink =
        DistributedVectorSink.bootstrapping(tmp, t3, params(), splitter(), policy(), METRIC, 2)) {
      sink.addAll(batch(0, 4, 0, 3L));
      sink.commit();
      sink.addAll(batch(1, 4, 4, 4L));
      sink.commit(); // 2nd commit → rotation
      sink.addAll(batch(2, 4, 8, 5L));
      sink.commit();
      assertThat(sink.committedCount()).isEqualTo(12L);
    }
  }

  @Test
  void closeIsIdempotent(@TempDir Path tmp) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    DistributedVectorSink sink =
        DistributedVectorSink.bootstrapping(tmp, t3, params(), splitter(), policy(), METRIC);
    sink.addAll(batch(0, 4, 0, 1L));
    sink.commit();
    sink.close();
    sink.close(); // no throw
    assertThatThrownBy(() -> sink.commit()).isInstanceOf(IllegalStateException.class);
  }
}
