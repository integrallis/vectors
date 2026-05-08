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
package com.integrallis.demos.studio;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.ingest.BatchPolicy;
import com.integrallis.vectors.ingest.BulkIngestor;
import com.integrallis.vectors.ingest.IngestDoc;
import com.integrallis.vectors.ingest.IngestResult;
import com.integrallis.vectors.ingest.cursor.InMemoryCursor;
import com.integrallis.vectors.ingest.embedders.LangChain4jEmbedder;
import com.integrallis.vectors.ingest.sinks.DistributedVectorSink;
import com.integrallis.vectors.ingest.sources.IterableSource;
import com.integrallis.vectors.ivf.ClusterSplitter;
import com.integrallis.vectors.ivf.IvfBuildParams;
import com.integrallis.vectors.ivf.TierPolicy;
import com.integrallis.vectors.storage.backend.S3StorageBackend;
import com.integrallis.vectors.storage.backend.StorageBackend;
import com.integrallis.vectors.studio.distributed.PrefixedStorageBackend;
import com.integrallis.vectors.studio.sidecart.ingest.SidecartWriterSink;
import com.integrallis.vectors.studio.sidecart.sources.H2SidecartSource;
import com.integrallis.vectors.studio.sidecart.sources.H2SidecartWriter;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import java.net.URI;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.api.io.TempDir;

/**
 * Live end-to-end test that drives the demo corpus through {@link BulkIngestor} into a real R2
 * bucket plus a temp-file H2 sidecart. Skipped when {@code .env} is missing the {@code
 * VECTORS_R2_*} keys.
 */
@Tag("integration")
@EnabledIf("isR2Configured")
class BulkIngestorR2IT {

  private static final R2Config CFG = R2Config.fromEnv();
  private static final SimilarityFunction METRIC = SimilarityFunction.COSINE;

  @SuppressWarnings("unused")
  static boolean isR2Configured() {
    return CFG != null;
  }

  @Test
  void ingestsDemoCorpusToR2AndH2(@TempDir Path tmp) throws Exception {
    String runPrefix = "ingest-it/" + UUID.randomUUID() + "/";
    StorageBackend root =
        S3StorageBackend.create(
            URI.create(CFG.endpoint), CFG.bucket, CFG.region, CFG.accessKey, CFG.secretKey);
    StorageBackend t3 = new PrefixedStorageBackend(root, runPrefix);
    Path walDir = tmp.resolve("wal");
    java.nio.file.Files.createDirectories(walDir);

    String h2Url = "jdbc:h2:" + tmp.resolve("sidecart").toAbsolutePath() + ";DB_CLOSE_DELAY=-1";
    H2SidecartWriter h2Writer =
        new H2SidecartWriter(h2Url, "sa", "", "DOCS", "ID", "TEXT", null, null);
    h2Writer.ensureSchema();

    DistributedVectorSink vectorSink =
        DistributedVectorSink.bootstrapping(
            walDir,
            t3,
            new IvfBuildParams(8, 30, 0f, false, 42L, 0).withPq(16, 256, -1f),
            new ClusterSplitter(10_000, 30, 42L),
            new TierPolicy(5, 2),
            METRIC);

    List<Doc> corpus = Corpus.realistic();
    List<IngestDoc> docs = corpus.stream().map(d -> IngestDoc.text(d.id(), d.text())).toList();
    IngestResult r;
    try (BulkIngestor ing =
        BulkIngestor.builder()
            .vectorSink(vectorSink)
            .sidecartSink(SidecartWriterSink.forH2(h2Writer))
            .embedder(new LangChain4jEmbedder(new AllMiniLmL6V2EmbeddingModel()))
            .batchPolicy(new BatchPolicy(64, 8L * 1024L * 1024L, Duration.ofSeconds(2)))
            .cursor(new InMemoryCursor())
            .embeddingConcurrency(2)
            .queueCapacity(16)
            .build()) {
      r = ing.ingest(IterableSource.of("demo-corpus", docs));
    }

    assertThat(r.docsCommitted()).isEqualTo(corpus.size());
    assertThat(r.firstError()).isEmpty();

    H2SidecartSource source =
        new H2SidecartSource(h2Url, "sa", "", "DOCS", "ID", "TEXT", null, null);
    for (Doc d : corpus) {
      assertThat(source.get(d.id()))
          .as("sidecart row for %s", d.id())
          .isPresent()
          .get()
          .extracting("text")
          .isEqualTo(d.text());
    }

    // R2 should have at least the routing index + per-cluster snapshots under the run prefix.
    List<String> keys = root.list(runPrefix);
    assertThat(keys).isNotEmpty();

    // Best-effort cleanup of the run prefix to keep the bucket tidy.
    for (String k : keys) {
      try {
        root.delete(k);
      } catch (Exception ignored) {
        // ignore — the @TempDir wal cleans up locally regardless.
      }
    }
  }
}
