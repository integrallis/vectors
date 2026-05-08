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
package com.integrallis.vectors.ivf;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.storage.backend.StorageBackend;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.MethodOrderer.OrderAnnotation;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.TestInstance.Lifecycle;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.api.io.TempDir;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;

/**
 * End-to-end integration test against a real Cloudflare R2 bucket.
 *
 * <p>Demonstrates the regular-user flow: embed a text corpus with a real sentence-transformer model
 * ({@code all-MiniLM-L6-v2}, 384-dim ONNX), build a {@link DistributedVectorCollection} backed by
 * R2 with product-quantization compression, run semantic similarity searches, add and commit a new
 * document, reopen the collection from R2 alone, and clean up.
 *
 * <p>Credentials are loaded from a gitignored {@code .env} at the repo root (see {@code
 * .env.example}). The whole class auto-skips when {@code VECTORS_R2_*} variables are missing, so it
 * is safe to leave on the default test classpath.
 *
 * <p>Tagged {@code @Tag("integration")} — run via:
 *
 * <pre>{@code ./gradlew :vectors-ivf:integrationTest --tests '*R2IT'}</pre>
 *
 * <p><b>Scope.</b> {@link DistributedVectorCollection} currently supports the Create/Read/reopen
 * subset of CRUD (build, add, commit, search, open). Row-level Update and Delete are Phase-2
 * features in {@code vectors-distributed-design.md} and are not exercised here.
 */
@Tag("integration")
@TestInstance(Lifecycle.PER_CLASS)
@TestMethodOrder(OrderAnnotation.class)
@EnabledIf("isR2Configured")
class DistributedVectorCollectionR2IT {

  /* Loaded once per class from .env at repo root (or any ancestor of CWD). */
  private static final R2Config CFG = R2Config.fromEnv();

  /** JUnit @EnabledIf hook: skip the whole class when R2 is not configured in .env. */
  @SuppressWarnings("unused")
  static boolean isR2Configured() {
    return CFG != null;
  }

  /** Per-run key prefix so multiple invocations against the same bucket never collide. */
  private static final String RUN_PREFIX = "r2-it/" + UUID.randomUUID() + "/";

  private static final SimilarityFunction METRIC = SimilarityFunction.COSINE;
  private static final int DIM = 384;
  private static final long SEED = 42L;

  private S3Client s3;
  private StorageBackend t3Backend;
  private EmbeddingModel embedder;
  private List<Doc> corpus;

  /** Shared WAL dir used across the @Order-ed tests so reopen sees the same state. */
  @TempDir Path walDir;

  // ─── lifecycle ─────────────────────────────────────────────────────────────

  @BeforeAll
  void connectAndEmbed() {
    s3 =
        S3Client.builder()
            .region(Region.of(CFG.region))
            .endpointOverride(URI.create(CFG.endpoint))
            .credentialsProvider(
                StaticCredentialsProvider.create(
                    AwsBasicCredentials.create(CFG.accessKey, CFG.secretKey)))
            .forcePathStyle(true)
            .build();
    t3Backend = new PrefixedS3Backend(s3, CFG.bucket, RUN_PREFIX);

    // First call downloads ~90 MB of ONNX weights to ~/.cache/langchain4j (one time).
    embedder = new AllMiniLmL6V2EmbeddingModel();
    corpus = Corpus.realistic();
  }

  @AfterAll
  void cleanupBucket() throws IOException {
    if (t3Backend == null) return;
    int deleted = 0;
    for (String key : t3Backend.list("")) {
      t3Backend.delete(key);
      deleted++;
    }
    System.out.printf("[R2 IT] Deleted %d objects under %s%n", deleted, RUN_PREFIX);
    s3.close();
  }

  // ─── helpers ───────────────────────────────────────────────────────────────

  private float[][] embedAll(List<Doc> docs) {
    List<TextSegment> segments = docs.stream().map(d -> TextSegment.from(d.text())).toList();
    var response = embedder.embedAll(segments);
    float[][] out = new float[response.content().size()][];
    for (int i = 0; i < out.length; i++) out[i] = response.content().get(i).vector();
    return out;
  }

  private float[] embedOne(String text) {
    return embedder.embed(text).content().vector();
  }

  private String[] idsFrom(List<Doc> docs) {
    return docs.stream().map(Doc::id).toArray(String[]::new);
  }

  private DistributedVectorCollection build(float[][] vecs, String[] ids) throws IOException {
    // PQ-16: 384 / 16 = 24-dim subspaces, 256 centroids → ~16 bytes/vector codes.
    IvfBuildParams params = new IvfBuildParams(8, 30, 0f, false, SEED, 0).withPq(16, 256, -1f);
    ClusterSplitter splitter = new ClusterSplitter(10_000, 30, SEED);
    TierPolicy policy = new TierPolicy(5, 2);
    return DistributedVectorCollection.build(
        vecs, ids, METRIC, params, splitter, policy, walDir, t3Backend);
  }

  // ─── tests ─────────────────────────────────────────────────────────────────

  @Test
  @Order(1)
  @DisplayName(
      "vectorize text → quantize with PQ-16 → store in R2 → semantic search returns same-topic docs")
  void create_and_search() throws IOException {
    float[][] embeddings = embedAll(corpus);
    String[] docIds = idsFrom(corpus);

    try (var col = build(embeddings, docIds)) {
      assertThat(col.size()).isEqualTo(corpus.size());

      // T3 must contain the routing index plus at least one cluster snapshot in R2.
      List<String> keys = t3Backend.list("");
      assertThat(keys).anyMatch(k -> k.endsWith("routing-index"));
      assertThat(keys.stream().filter(k -> k.contains("cluster-")).count()).isGreaterThan(0);

      // Semantic search: a food query should return food docs in the top-3.
      float[] q = embedOne("What sushi rolls do you recommend for dinner?");
      List<IvfHit> hits = col.search(q, 3, 4);
      assertThat(hits).isNotEmpty();
      List<String> topTopics = hits.stream().map(h -> Corpus.topicOf(h.id())).toList();
      assertThat(topTopics).contains("food");

      // Programming query should return programming docs.
      float[] q2 = embedOne("How do I write a recursive function in Java?");
      List<IvfHit> hits2 = col.search(q2, 3, 4);
      assertThat(hits2.stream().map(h -> Corpus.topicOf(h.id()))).contains("programming");
    }
  }

  @Test
  @Order(2)
  @DisplayName("incremental add + commit persists the new doc to R2 and is searchable")
  void add_commit_search() throws IOException {
    float[][] embeddings = embedAll(corpus);
    String[] docIds = idsFrom(corpus);

    try (var col = build(embeddings, docIds)) {
      String newId = "extra-001";
      String newText = "Quantum entanglement is a hallmark of modern physics.";
      col.add(newId, embedOne(newText));
      col.commit();
      assertThat(col.size()).isEqualTo(corpus.size() + 1);

      List<IvfHit> hits = col.search(embedOne("physics and quantum mechanics"), 5, 4);
      assertThat(hits.stream().map(IvfHit::id)).contains(newId);
    }
  }

  @Test
  @Order(3)
  @DisplayName("close + reopen restores collection state from R2 + WAL alone")
  void reopen_from_r2() throws IOException {
    String probeId = corpus.get(0).id();
    float[] probeVec = embedOne(corpus.get(0).text());

    // Build, close.
    try (var col = build(embedAll(corpus), idsFrom(corpus))) {
      assertThat(col.size()).isEqualTo(corpus.size());
    }

    // Reopen using only walDir + R2 backend; no in-memory state carried over.
    try (var col =
        DistributedVectorCollection.open(walDir, METRIC, new TierPolicy(5, 2), t3Backend)) {
      assertThat(col.size()).isEqualTo(corpus.size());
      List<IvfHit> hits = col.search(probeVec, 1, 4);
      assertThat(hits).isNotEmpty();
      assertThat(hits.get(0).id()).isEqualTo(probeId);
    }
  }
}
