package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.storage.FileFormat;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * End-to-end tests for Step 4d quantization support across all 6 quantizer kinds (SQ8, SQ4, PQ, BQ,
 * RABITQ, NVQ) with both HNSW and VAMANA indexes in in-memory and persistent modes.
 */
class VectorDbQuantizationTest {

  private static final long SEED = 42L;
  // Use 64-dim so RaBitQ (which pads to 64-multiple) doesn't pad, and PQ subspaces divide evenly.
  private static final int DIM = 64;

  private static float[] randomVector(Random rng) {
    float[] v = new float[DIM];
    for (int i = 0; i < DIM; i++) {
      v[i] = rng.nextFloat() - 0.5f;
    }
    return v;
  }

  private static List<Document> generateDocs(int count, long seed) {
    Random rng = new Random(seed);
    List<Document> out = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      out.add(new Document("doc-" + i, randomVector(rng), "text-" + i, null));
    }
    return out;
  }

  private static Set<String> bruteForceTopKIds(List<Document> docs, float[] query, int k) {
    SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;
    String[] ids = new String[docs.size()];
    float[] scores = new float[docs.size()];
    for (int i = 0; i < docs.size(); i++) {
      ids[i] = docs.get(i).id();
      scores[i] = sim.compare(query, docs.get(i).vector());
    }
    Set<String> top = new HashSet<>();
    boolean[] used = new boolean[docs.size()];
    for (int r = 0; r < Math.min(k, docs.size()); r++) {
      int best = -1;
      for (int i = 0; i < docs.size(); i++) {
        if (used[i]) continue;
        if (best == -1 || scores[i] > scores[best]) best = i;
      }
      used[best] = true;
      top.add(ids[best]);
    }
    return top;
  }

  private static double recall(Set<String> groundTruth, SearchResult result) {
    Set<String> returned = new HashSet<>();
    for (SearchResult.Hit h : result.hits()) {
      returned.add(h.id());
    }
    long overlap = groundTruth.stream().filter(returned::contains).count();
    return (double) overlap / groundTruth.size();
  }

  // ---------------------------------------------------------------------------
  // In-memory HNSW + Quantization
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class InMemoryHnswQuantized {

    @Test
    void sq8ProducesRecallAboveThreshold() {
      List<Document> docs = generateDocs(500, SEED);
      float[] query = randomVector(new Random(999L));

      VectorCollection coll =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .quantizer(QuantizerKind.SQ8)
              .build();
      coll.addAll(docs);
      coll.commit();

      // Search with two-pass (overQueryFactor > 1)
      SearchResult result =
          coll.search(
              SearchRequest.builder(query, 10).searchListSize(100).overQueryFactor(2.0f).build());

      Set<String> gt = bruteForceTopKIds(docs, query, 10);
      double r = recall(gt, result);
      assertThat(r).as("SQ8 HNSW recall").isGreaterThanOrEqualTo(0.70);
      coll.close();
    }

    @Test
    void overQueryFactorImprovesRecallOrMaintainsIt() {
      List<Document> docs = generateDocs(500, SEED);
      float[] query = randomVector(new Random(999L));

      VectorCollection coll =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .quantizer(QuantizerKind.SQ8)
              .build();
      coll.addAll(docs);
      coll.commit();

      Set<String> gt = bruteForceTopKIds(docs, query, 10);

      // Single-pass (overQueryFactor=1.0, no two-pass)
      SearchResult single =
          coll.search(
              SearchRequest.builder(query, 10).searchListSize(100).overQueryFactor(1.0f).build());

      // Two-pass with overQueryFactor=3.0
      SearchResult twoPass =
          coll.search(
              SearchRequest.builder(query, 10).searchListSize(100).overQueryFactor(3.0f).build());

      double recallSingle = recall(gt, single);
      double recallTwoPass = recall(gt, twoPass);

      // Two-pass should be at least as good as single-pass
      assertThat(recallTwoPass)
          .as("two-pass recall >= single-pass recall")
          .isGreaterThanOrEqualTo(recallSingle);
      coll.close();
    }

    @Test
    void sq4ReturnsResults() {
      List<Document> docs = generateDocs(200, SEED);
      float[] query = randomVector(new Random(999L));

      VectorCollection coll =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .quantizer(QuantizerKind.SQ4)
              .build();
      coll.addAll(docs);
      coll.commit();

      SearchResult result =
          coll.search(
              SearchRequest.builder(query, 10).searchListSize(100).overQueryFactor(2.0f).build());
      assertThat(result.hits()).isNotEmpty();
      assertThat(result.hits().size()).isLessThanOrEqualTo(10);
      coll.close();
    }

    @Test
    void pqReturnsResults() {
      List<Document> docs = generateDocs(200, SEED);
      float[] query = randomVector(new Random(999L));

      VectorCollection coll =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .quantizer(QuantizerKind.PQ)
              .pqSubspaces(8)
              .pqClusters(16)
              .build();
      coll.addAll(docs);
      coll.commit();

      SearchResult result =
          coll.search(
              SearchRequest.builder(query, 10).searchListSize(100).overQueryFactor(2.0f).build());
      assertThat(result.hits()).isNotEmpty();
      coll.close();
    }

    @Test
    void bqReturnsResults() {
      List<Document> docs = generateDocs(200, SEED);
      float[] query = randomVector(new Random(999L));

      VectorCollection coll =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .quantizer(QuantizerKind.BQ)
              .bqMode(true) // BBQ
              .build();
      coll.addAll(docs);
      coll.commit();

      SearchResult result =
          coll.search(
              SearchRequest.builder(query, 10).searchListSize(100).overQueryFactor(2.0f).build());
      assertThat(result.hits()).isNotEmpty();
      coll.close();
    }

    @Test
    void rabitqReturnsResults() {
      List<Document> docs = generateDocs(200, SEED);
      float[] query = randomVector(new Random(999L));

      VectorCollection coll =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .quantizer(QuantizerKind.RABITQ)
              .build();
      coll.addAll(docs);
      coll.commit();

      SearchResult result =
          coll.search(
              SearchRequest.builder(query, 10).searchListSize(100).overQueryFactor(2.0f).build());
      assertThat(result.hits()).isNotEmpty();
      coll.close();
    }

    @Test
    void nvqReturnsResults() {
      List<Document> docs = generateDocs(200, SEED);
      float[] query = randomVector(new Random(999L));

      VectorCollection coll =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .quantizer(QuantizerKind.NVQ)
              .nvqSubvectors(8)
              .build();
      coll.addAll(docs);
      coll.commit();

      SearchResult result =
          coll.search(
              SearchRequest.builder(query, 10).searchListSize(100).overQueryFactor(2.0f).build());
      assertThat(result.hits()).isNotEmpty();
      coll.close();
    }
  }

  // ---------------------------------------------------------------------------
  // In-memory Vamana + Quantization
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class InMemoryVamanaQuantized {

    @Test
    void sq8ProducesRecallAboveThreshold() {
      List<Document> docs = generateDocs(500, SEED);
      float[] query = randomVector(new Random(999L));

      VectorCollection coll =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .quantizer(QuantizerKind.SQ8)
              .build();
      coll.addAll(docs);
      coll.commit();

      SearchResult result =
          coll.search(
              SearchRequest.builder(query, 10).searchListSize(128).overQueryFactor(2.0f).build());

      Set<String> gt = bruteForceTopKIds(docs, query, 10);
      double r = recall(gt, result);
      assertThat(r).as("SQ8 Vamana recall").isGreaterThanOrEqualTo(0.70);
      coll.close();
    }

    @Test
    void bqReturnsResults() {
      List<Document> docs = generateDocs(200, SEED);
      float[] query = randomVector(new Random(999L));

      VectorCollection coll =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .quantizer(QuantizerKind.BQ)
              .build();
      coll.addAll(docs);
      coll.commit();

      SearchResult result =
          coll.search(
              SearchRequest.builder(query, 10).searchListSize(128).overQueryFactor(2.0f).build());
      assertThat(result.hits()).isNotEmpty();
      coll.close();
    }

    @Test
    void pqReturnsResults() {
      List<Document> docs = generateDocs(200, SEED);
      float[] query = randomVector(new Random(999L));

      VectorCollection coll =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .quantizer(QuantizerKind.PQ)
              .pqSubspaces(8)
              .pqClusters(16)
              .build();
      coll.addAll(docs);
      coll.commit();

      SearchResult result =
          coll.search(
              SearchRequest.builder(query, 10).searchListSize(128).overQueryFactor(2.0f).build());
      assertThat(result.hits()).isNotEmpty();
      coll.close();
    }
  }

  // ---------------------------------------------------------------------------
  // Persistent HNSW + Quantization round-trip
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class PersistentHnswQuantized {

    @Test
    void sq8RoundTrips(@TempDir Path dir) {
      assertQuantizedPersistentHnswRoundTrip(dir, QuantizerKind.SQ8, null);
    }

    @Test
    void sq4RoundTrips(@TempDir Path dir) {
      assertQuantizedPersistentHnswRoundTrip(dir, QuantizerKind.SQ4, null);
    }

    @Test
    void pqRoundTrips(@TempDir Path dir) {
      VectorCollectionBuilder b =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .quantizer(QuantizerKind.PQ)
              .pqSubspaces(8)
              .pqClusters(16);
      assertQuantizedPersistentRoundTrip(dir, b);
    }

    @Test
    void bqRoundTrips(@TempDir Path dir) {
      assertQuantizedPersistentHnswRoundTrip(dir, QuantizerKind.BQ, null);
    }

    @Test
    void rabitqRoundTrips(@TempDir Path dir) {
      assertQuantizedPersistentHnswRoundTrip(dir, QuantizerKind.RABITQ, null);
    }

    @Test
    void nvqRoundTrips(@TempDir Path dir) {
      VectorCollectionBuilder b =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .quantizer(QuantizerKind.NVQ)
              .nvqSubvectors(8);
      assertQuantizedPersistentRoundTrip(dir, b);
    }

    private void assertQuantizedPersistentHnswRoundTrip(
        Path dir, QuantizerKind kind, QuantizerParams params) {
      VectorCollectionBuilder b =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .quantizer(kind);
      assertQuantizedPersistentRoundTrip(dir, b);
    }
  }

  // ---------------------------------------------------------------------------
  // Persistent Vamana + Quantization round-trip
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class PersistentVamanaQuantized {

    @Test
    void sq8RoundTrips(@TempDir Path dir) {
      VectorCollectionBuilder b =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .quantizer(QuantizerKind.SQ8);
      assertQuantizedPersistentRoundTrip(dir, b);
    }

    @Test
    void pqRoundTrips(@TempDir Path dir) {
      VectorCollectionBuilder b =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .quantizer(QuantizerKind.PQ)
              .pqSubspaces(8)
              .pqClusters(16);
      assertQuantizedPersistentRoundTrip(dir, b);
    }

    @Test
    void bqRoundTrips(@TempDir Path dir) {
      VectorCollectionBuilder b =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .quantizer(QuantizerKind.BQ);
      assertQuantizedPersistentRoundTrip(dir, b);
    }

    @Test
    void rabitqRoundTrips(@TempDir Path dir) {
      VectorCollectionBuilder b =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .quantizer(QuantizerKind.RABITQ);
      assertQuantizedPersistentRoundTrip(dir, b);
    }

    @Test
    void nvqRoundTrips(@TempDir Path dir) {
      VectorCollectionBuilder b =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .quantizer(QuantizerKind.NVQ)
              .nvqSubvectors(8);
      assertQuantizedPersistentRoundTrip(dir, b);
    }

    @Test
    void sq4RoundTrips(@TempDir Path dir) {
      VectorCollectionBuilder b =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .quantizer(QuantizerKind.SQ4);
      assertQuantizedPersistentRoundTrip(dir, b);
    }
  }

  // ---------------------------------------------------------------------------
  // FLAT + Quantization blocked
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class FlatQuantizationBlocked {

    @Test
    void flatWithSq8ThrowsUnsupported() {
      assertThatThrownBy(
              () ->
                  VectorCollection.builder()
                      .dimension(DIM)
                      .metric(SimilarityFunction.EUCLIDEAN)
                      .indexType(IndexType.FLAT)
                      .quantizer(QuantizerKind.SQ8)
                      .build())
          .isInstanceOf(UnsupportedOperationException.class)
          .hasMessageContaining("FLAT");
    }

    @Test
    void flatWithPqThrowsUnsupported() {
      assertThatThrownBy(
              () ->
                  VectorCollection.builder()
                      .dimension(DIM)
                      .metric(SimilarityFunction.EUCLIDEAN)
                      .indexType(IndexType.FLAT)
                      .quantizer(QuantizerKind.PQ)
                      .build())
          .isInstanceOf(UnsupportedOperationException.class)
          .hasMessageContaining("FLAT");
    }

    @Test
    void flatWithNoneIsAllowed() {
      VectorCollection coll =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.FLAT)
              .quantizer(QuantizerKind.NONE)
              .build();
      assertThat(coll.size()).isZero();
      coll.close();
    }
  }

  // ---------------------------------------------------------------------------
  // Crash recovery: corrupt quantized.bin
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class SimulatedKill9Quantized {

    @Test
    void corruptQuantizedBinFallsBackToPriorGeneration(@TempDir Path dir) throws IOException {
      Path storageRoot = dir.resolve("store");
      Files.createDirectories(storageRoot);

      // Gen-0 is the empty bootstrap. Gen-1: 100 docs, quantized HNSW.
      VectorCollection coll =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .quantizer(QuantizerKind.SQ8)
              .storagePath(storageRoot)
              .build();
      List<Document> docs1 = generateDocs(100, SEED);
      coll.addAll(docs1);
      coll.commit();
      assertThat(coll.size()).isEqualTo(100);
      coll.close();

      // Gen-2: add 50 more docs (total 150)
      coll =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .quantizer(QuantizerKind.SQ8)
              .storagePath(storageRoot)
              .build();
      List<Document> docs2 = generateDocs(50, SEED + 100);
      // Rename to avoid duplicates
      List<Document> renamedDocs2 = new ArrayList<>();
      for (int i = 0; i < docs2.size(); i++) {
        Document d = docs2.get(i);
        renamedDocs2.add(new Document("extra-" + i, d.vector(), d.text(), d.metadata()));
      }
      coll.addAll(renamedDocs2);
      coll.commit();
      assertThat(coll.size()).isEqualTo(150);
      coll.close();

      // Corrupt gen-2's quantized.bin by zeroing it out.
      // Generation numbering: gen-0 = empty bootstrap, gen-1 = first commit (100 docs),
      // gen-2 = second commit (150 docs). CURRENT → 2.
      Path gen2Dir = storageRoot.resolve(FileFormat.generationDirName(2L));
      Path quantizedFile = gen2Dir.resolve(FileFormat.QUANTIZED_FILE);
      assertThat(Files.exists(quantizedFile)).as("quantized.bin exists in gen-2").isTrue();

      byte[] zeros = new byte[(int) Files.size(quantizedFile)];
      Files.write(quantizedFile, zeros);

      // Reopen should fall back to gen-1 (100 docs), NOT gen-2 (150 docs)
      coll =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .quantizer(QuantizerKind.SQ8)
              .storagePath(storageRoot)
              .build();
      assertThat(coll.size()).as("fell back to gen-1").isEqualTo(100);
      coll.close();
    }
  }

  // ---------------------------------------------------------------------------
  // Concurrency: concurrent readers with quantized persistent HNSW
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("slow")
  class ConcurrencyQuantized {

    @Test
    void concurrentReadersWithQuantizedHnsw(@TempDir Path dir) {
      assertTimeoutPreemptively(
          Duration.ofSeconds(60),
          () -> {
            Path storageRoot = dir.resolve("store");
            Files.createDirectories(storageRoot);

            VectorCollection coll =
                VectorCollection.builder()
                    .dimension(DIM)
                    .metric(SimilarityFunction.EUCLIDEAN)
                    .indexType(IndexType.HNSW)
                    .quantizer(QuantizerKind.SQ8)
                    .storagePath(storageRoot)
                    .build();

            int nReaders = 4;
            int nCommits = 10;
            int docsPerCommit = 30;

            AtomicBoolean stop = new AtomicBoolean(false);
            AtomicReference<Throwable> readerError = new AtomicReference<>();
            AtomicInteger searchCount = new AtomicInteger(0);

            // Start readers
            ExecutorService readers = Executors.newFixedThreadPool(nReaders);
            CountDownLatch started = new CountDownLatch(nReaders);
            for (int r = 0; r < nReaders; r++) {
              final int readerIdx = r;
              readers.submit(
                  () -> {
                    Random rng = new Random(SEED + readerIdx + 100L);
                    started.countDown();
                    while (!stop.get()) {
                      try {
                        float[] q = randomVector(rng);
                        SearchResult sr =
                            coll.search(
                                SearchRequest.builder(q, 5)
                                    .searchListSize(50)
                                    .overQueryFactor(2.0f)
                                    .build());
                        // Just verify it doesn't throw and returns valid results
                        assertThat(sr.hits().size()).isLessThanOrEqualTo(5);
                        searchCount.incrementAndGet();
                      } catch (Throwable t) {
                        if (!stop.get()) {
                          readerError.compareAndSet(null, t);
                        }
                        return;
                      }
                    }
                  });
            }

            started.await();

            // Writer: add docs and commit
            for (int c = 0; c < nCommits; c++) {
              List<Document> batch = new ArrayList<>(docsPerCommit);
              Random rng = new Random(SEED + c * 1000L);
              for (int i = 0; i < docsPerCommit; i++) {
                String id = "c" + c + "-d" + i;
                batch.add(new Document(id, randomVector(rng), null, null));
              }
              coll.addAll(batch);
              coll.commit();
            }

            stop.set(true);
            readers.shutdown();
            readers.awaitTermination(60, TimeUnit.SECONDS);

            assertThat(readerError.get()).as("no reader errors").isNull();
            assertThat(searchCount.get()).as("readers did searches").isGreaterThan(0);
            assertThat(coll.size()).isEqualTo(nCommits * docsPerCommit);
            coll.close();
          });
    }
  }

  // ---------------------------------------------------------------------------
  // Shared helper: persistent round-trip assertion
  // ---------------------------------------------------------------------------

  /**
   * Shared assertion for persistent round-trip tests. Builds a collection with 200 docs, commits,
   * closes, reopens, and verifies the same search results are returned (same IDs and close scores).
   */
  private void assertQuantizedPersistentRoundTrip(Path dir, VectorCollectionBuilder builder) {
    Path storageRoot = dir.resolve("store");
    try {
      Files.createDirectories(storageRoot);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }

    List<Document> docs = generateDocs(200, SEED);
    float[] query = randomVector(new Random(999L));

    // Build, add, commit, search
    VectorCollection coll = builder.storagePath(storageRoot).build();
    coll.addAll(docs);
    coll.commit();

    SearchResult beforeClose =
        coll.search(
            SearchRequest.builder(query, 10).searchListSize(100).overQueryFactor(2.0f).build());
    assertThat(beforeClose.hits()).isNotEmpty();

    int[] beforeIds =
        beforeClose.hits().stream().mapToInt(h -> Integer.parseInt(h.id().substring(4))).toArray();
    float[] beforeScores = new float[beforeClose.hits().size()];
    for (int i = 0; i < beforeScores.length; i++) {
      beforeScores[i] = beforeClose.hits().get(i).score();
    }

    coll.close();

    // Reopen and search again
    VectorCollection coll2 = builder.storagePath(storageRoot).build();
    assertThat(coll2.size()).isEqualTo(200);

    SearchResult afterReopen =
        coll2.search(
            SearchRequest.builder(query, 10).searchListSize(100).overQueryFactor(2.0f).build());
    assertThat(afterReopen.hits()).isNotEmpty();

    int[] afterIds =
        afterReopen.hits().stream().mapToInt(h -> Integer.parseInt(h.id().substring(4))).toArray();
    float[] afterScores = new float[afterReopen.hits().size()];
    for (int i = 0; i < afterScores.length; i++) {
      afterScores[i] = afterReopen.hits().get(i).score();
    }

    assertThat(afterIds).as("same result IDs after reopen").isEqualTo(beforeIds);
    for (int i = 0; i < beforeScores.length; i++) {
      assertThat(afterScores[i])
          .as("score[" + i + "] matches")
          .isCloseTo(beforeScores[i], org.assertj.core.data.Offset.offset(1e-4f));
    }

    coll2.close();
  }
}
