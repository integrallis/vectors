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

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.ingest.BatchPolicy;
import com.integrallis.vectors.ingest.BulkIngestor;
import com.integrallis.vectors.ingest.IngestDoc;
import com.integrallis.vectors.ingest.IngestResult;
import com.integrallis.vectors.ingest.RetryPolicy;
import com.integrallis.vectors.ingest.cursor.InMemoryCursor;
import com.integrallis.vectors.ingest.embedders.PrecomputedEmbedder;
import com.integrallis.vectors.ingest.sources.IterableSource;
import com.integrallis.vectors.ivf.ClusterSplitter;
import com.integrallis.vectors.ivf.DistributedVectorCollection;
import com.integrallis.vectors.ivf.IvfBuildParams;
import com.integrallis.vectors.ivf.IvfHit;
import com.integrallis.vectors.ivf.TierPolicy;
import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class DistributedVectorSinkPipelineTest {

  private static final int DIM = 8;

  private static List<IngestDoc> randomDocs(int n, long seed) {
    Random rng = new Random(seed);
    List<IngestDoc> out = new ArrayList<>(n);
    for (int i = 0; i < n; i++) {
      float[] v = new float[DIM];
      for (int d = 0; d < DIM; d++) v[d] = rng.nextFloat() * 2f - 1f;
      out.add(new IngestDoc("id-" + i, "doc-" + i, null, "text/plain", null, v));
    }
    return out;
  }

  @Test
  void bulkIngestorDrivesDistributedVectorSinkBootstrapping(@TempDir Path tmp) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    DistributedVectorSink sink =
        DistributedVectorSink.bootstrapping(
            tmp,
            t3,
            new IvfBuildParams(2, 30, 0f, false, 42L, 0),
            new ClusterSplitter(10_000, 30, 42L),
            new TierPolicy(5, 2),
            SimilarityFunction.COSINE);
    InMemoryCursor cursor = new InMemoryCursor();
    List<IngestDoc> docs = randomDocs(40, 13L);
    try (BulkIngestor ing =
        BulkIngestor.builder()
            .vectorSink(sink)
            .embedder(new PrecomputedEmbedder(DIM))
            .batchPolicy(new BatchPolicy(10, 1024L * 1024L, Duration.ofSeconds(5)))
            .cursor(cursor)
            .retryPolicy(noRetry())
            .embeddingConcurrency(2)
            .queueCapacity(16)
            .build()) {
      IngestResult r = ing.ingest(IterableSource.of("dvs", docs));
      assertThat(r.docsCommitted()).isEqualTo(40L);
      assertThat(r.batchesCommitted()).isEqualTo(4L);
      assertThat(r.lastCursor()).isEqualTo(39L);
    }
    DistributedVectorCollection col = sink.collection();
    assertThat(col.size()).isEqualTo(40);
    // Probe the collection with the first vector and confirm we hit its own id.
    float[] q = docs.get(0).precomputedVector();
    List<IvfHit> hits = col.search(q, 1, 2);
    assertThat(hits).isNotEmpty();
    assertThat(hits.get(0).id()).isEqualTo("id-0");
    assertThat(cursor.load("dvs")).isEqualTo(39L);
  }

  private static RetryPolicy noRetry() {
    return new RetryPolicy(
        1, Duration.ofMillis(1), Duration.ofMillis(1), 0.0, new Random(0L), ms -> {});
  }
}
