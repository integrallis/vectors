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
package com.integrallis.vectors.studio.sidecart.ingest;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.ingest.BatchPolicy;
import com.integrallis.vectors.ingest.BulkIngestor;
import com.integrallis.vectors.ingest.IngestDoc;
import com.integrallis.vectors.ingest.IngestResult;
import com.integrallis.vectors.ingest.cursor.InMemoryCursor;
import com.integrallis.vectors.ingest.embedders.PrecomputedEmbedder;
import com.integrallis.vectors.ingest.sinks.DistributedVectorSink;
import com.integrallis.vectors.ingest.sources.IterableSource;
import com.integrallis.vectors.ivf.ClusterSplitter;
import com.integrallis.vectors.ivf.IvfBuildParams;
import com.integrallis.vectors.ivf.IvfHit;
import com.integrallis.vectors.ivf.TierPolicy;
import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.studio.sidecart.SidecartRecord;
import com.integrallis.vectors.studio.sidecart.sources.H2SidecartSource;
import com.integrallis.vectors.studio.sidecart.sources.H2SidecartWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * 5 000-document end-to-end test that drives {@link BulkIngestor} through {@link
 * DistributedVectorSink} (against {@link HeapStorageBackend}) plus {@link SidecartWriterSink} (H2
 * in-memory), then performs a strict-idempotency rerun: a second ingest with a cursor positioned
 * past the last committed offset must commit zero docs and leave both backends unchanged.
 */
@Tag("integration")
class DistributedVectorSinkH2IT {

  private static final int DIM = 32;
  private static final int N = 5_000;
  private static final SimilarityFunction METRIC = SimilarityFunction.COSINE;
  private static final String SOURCE_NAME = "synth-5k";

  @Test
  void ingests5kDocsAcrossSinksAndIsIdempotentOnRerun(@TempDir Path tmp) throws IOException {
    List<IngestDoc> docs = synthDocs(N, 0xC0FFEEL);
    HeapStorageBackend t3 = new HeapStorageBackend();
    Path walDir = tmp.resolve("wal");
    java.nio.file.Files.createDirectories(walDir);

    String h2Url = "jdbc:h2:mem:dvsh2_" + UUID.randomUUID() + ";DB_CLOSE_DELAY=-1";
    H2SidecartWriter h2Writer =
        new H2SidecartWriter(h2Url, "sa", "", "DOCS", "ID", "TEXT", null, null);
    h2Writer.ensureSchema();

    InMemoryCursor cursor = new InMemoryCursor();
    BatchPolicy bp = new BatchPolicy(200, 8L * 1024L * 1024L, Duration.ofSeconds(5));

    // ── First run: bootstrap-build + commit all 5 000 docs ──────────────────
    IngestResult first;
    try (DistributedVectorSink vectorSink =
            DistributedVectorSink.bootstrapping(
                walDir,
                t3,
                new IvfBuildParams(32, 20, 0f, false, 42L, 0),
                new ClusterSplitter(10_000, 20, 42L),
                new TierPolicy(5, 2),
                METRIC);
        BulkIngestor ing =
            BulkIngestor.builder()
                .vectorSink(vectorSink)
                .sidecartSink(SidecartWriterSink.forH2(h2Writer))
                .embedder(new PrecomputedEmbedder(DIM))
                .batchPolicy(bp)
                .cursor(cursor)
                .embeddingConcurrency(2)
                .queueCapacity(8)
                .build()) {
      first = ing.ingest(IterableSource.of(SOURCE_NAME, docs));

      assertThat(first.docsCommitted()).isEqualTo(N);
      assertThat(first.firstError()).isEmpty();
      assertThat(first.lastCursor()).isEqualTo(N - 1L);
      assertThat(vectorSink.collection().size()).isEqualTo(N);

      // T3 must contain at least the routing index plus per-cluster snapshots.
      List<String> keys = t3.list("");
      assertThat(keys).anyMatch(k -> k.endsWith("routing-index"));
      assertThat(keys.stream().filter(k -> k.contains("cluster-")).count()).isPositive();

      // Self-search probe on a sample doc must return its own id at top-1.
      int probe = 1234;
      List<IvfHit> hits = vectorSink.collection().search(docs.get(probe).precomputedVector(), 1, 4);
      assertThat(hits).isNotEmpty();
      assertThat(hits.get(0).id()).isEqualTo(docs.get(probe).id());
    }

    H2SidecartSource h2Source =
        new H2SidecartSource(h2Url, "sa", "", "DOCS", "ID", "TEXT", null, null);
    SidecartRecord row = h2Source.get(docs.get(42).id()).orElseThrow();
    assertThat(row.text()).isEqualTo(docs.get(42).text());

    // ── Rerun: open existing collection, resume past lastCursor → 0 commits ─
    long resumeOffset = cursor.load(SOURCE_NAME) + 1L;
    H2SidecartWriter h2Writer2 =
        new H2SidecartWriter(h2Url, "sa", "", "DOCS", "ID", "TEXT", null, null);
    try (DistributedVectorSink reopened =
            DistributedVectorSink.opening(walDir, t3, METRIC, new TierPolicy(5, 2));
        BulkIngestor ing =
            BulkIngestor.builder()
                .vectorSink(reopened)
                .sidecartSink(SidecartWriterSink.forH2(h2Writer2))
                .embedder(new PrecomputedEmbedder(DIM))
                .batchPolicy(bp)
                .cursor(cursor)
                .embeddingConcurrency(2)
                .queueCapacity(8)
                .build()) {
      IngestResult second = ing.ingest(IterableSource.resuming(SOURCE_NAME, docs, resumeOffset));
      assertThat(second.docsCommitted()).isZero();
      assertThat(second.batchesCommitted()).isZero();
      assertThat(second.firstError()).isEmpty();
      assertThat(reopened.collection().size()).isEqualTo(N);
    }
  }

  private static List<IngestDoc> synthDocs(int n, long seed) {
    Random rng = new Random(seed);
    List<IngestDoc> out = new ArrayList<>(n);
    for (int i = 0; i < n; i++) {
      float[] v = new float[DIM];
      for (int d = 0; d < DIM; d++) v[d] = rng.nextFloat() * 2f - 1f;
      out.add(new IngestDoc("doc-" + i, "synthetic content #" + i, null, "text/plain", null, v));
    }
    return out;
  }
}
