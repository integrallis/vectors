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
import com.integrallis.vectors.core.SimilarityFunction;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * A commit must not allocate in proportion to the whole collection.
 *
 * <p>Every commit used to build the successor generation in memory twice over — a {@code byte[]} of
 * the finished {@code vectors.bin} and a {@code float[][]} of every live vector — so the heap a
 * commit needed grew with the collection. An ingest of 460,000 documents at 512 dimensions died
 * with {@link OutOfMemoryError} inside the commit, and the {@code byte[]} also capped {@code
 * vectors.bin} at 2 GiB, about a million documents at that width, whatever the host had.
 *
 * <p>These tests are sized so that the old behaviour exceeds this JVM's heap while the collection
 * itself comfortably fits on disk, which is the situation that broke.
 */
class CommitMemoryFootprintTest {

  private static final int DIMENSION = 512;
  private static final int BATCH = 2_000;

  /**
   * A commit must not allocate a copy of the collection.
   *
   * <p>The successor generation used to be built in memory twice over — a {@code byte[]} of the
   * finished {@code vectors.bin} and a {@code float[][]} of every live vector — which at 512
   * dimensions is about 2 KiB per document each, so a commit allocated upwards of 4 KiB per
   * document held. That is what killed a 630,000-document ingest with {@link OutOfMemoryError} at
   * 460,000.
   *
   * <p>Measured rather than inferred from a heap limit: the test JVM's heap is large enough that an
   * OutOfMemoryError-based test passes with the defect present.
   *
   * <p>What this does <b>not</b> claim is that commit allocation is now independent of the
   * collection. It is not: roughly a kilobyte per document remains, from the HNSW graph object that
   * an append rebuilds and from the document list the commit assembles to rewrite {@code
   * metadata.bin} and {@code idmap.bin}. Removing those needs an in-place growable graph and
   * delta-written payloads respectively, neither of which this change attempts. The bound below
   * sits above what remains and below what the two full copies cost.
   */
  @Test
  void aCommitDoesNotAllocateACopyOfTheCollection(@TempDir Path dir) {
    int count = 100_000;
    long allocated = allocationOfFinalCommit(dir.resolve("hundredk"), count);
    double perDocument = allocated / (double) count;

    System.out.printf(
        "commit allocation at %,d documents: %,d bytes, %.0f per document%n",
        count, allocated, perDocument);

    // One stride is 2,048 bytes at this width. The old path allocated two full copies, so more than
    // two strides per document; what remains is about half a stride.
    assertThat(perDocument)
        .as("bytes allocated per document by one commit (total %,d)", allocated)
        .isLessThan(2_048.0);
  }

  /** Bytes allocated on this thread by the last commit of an ingest of {@code count} documents. */
  private static long allocationOfFinalCommit(Path storage, int count) {
    try (VectorCollection collection =
        VectorCollection.builder()
            .dimension(DIMENSION)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.HNSW)
            .storagePath(storage.toAbsolutePath())
            .build()) {

      SplittableRandom random = new SplittableRandom(99);
      List<Document> batch = new ArrayList<>(BATCH);
      for (int i = 0; i < count; i++) {
        batch.add(Document.of("v" + i, unitVector(random)));
        if (batch.size() == BATCH) {
          collection.addAll(batch);
          collection.commit();
          batch.clear();
        }
      }
      // One more batch, and measure only its commit.
      for (int i = 0; i < BATCH; i++) {
        batch.add(Document.of("tail" + i, unitVector(random)));
      }
      collection.addAll(batch);

      com.sun.management.ThreadMXBean threads =
          (com.sun.management.ThreadMXBean)
              java.lang.management.ManagementFactory.getThreadMXBean();
      long id = Thread.currentThread().threadId();
      long before = threads.getThreadAllocatedBytes(id);
      collection.commit();
      long after = threads.getThreadAllocatedBytes(id);

      assertThat(collection.size()).isEqualTo(count + BATCH);
      return after - before;
    }
  }

  /** The streamed metadata image must be byte-for-byte what the buffered writer produces. */
  @Test
  void streamedMetadataEqualsTheBufferedWriter() throws IOException {
    SplittableRandom random = new SplittableRandom(3);
    List<Document> documents = new ArrayList<>();
    for (int i = 0; i < 5_000; i++) {
      documents.add(
          new Document(
              "id-" + i,
              unitVector(random),
              "document " + i + " with some stored text of a realistic length for a corpus entry",
              java.util.Map.of(
                  "category", com.integrallis.vectors.core.MetadataValue.of("c" + (i % 14)),
                  "n", com.integrallis.vectors.core.MetadataValue.of(i))));
    }
    byte[] buffered =
        com.integrallis.vectors.db.storage.MappedMetadataStore.Writer.toBytes(documents);
    var image = new com.integrallis.vectors.db.storage.MappedMetadataStore.Writer.Image(documents);

    assertThat(image.length()).isEqualTo(buffered.length);
    assertThat(image.crc32())
        .isEqualTo(com.integrallis.vectors.db.storage.Checksums.ofBytes(buffered));

    Path tmp = Files.createTempFile("metadata-image", ".bin");
    try {
      image.writeTo(tmp);
      assertThat(Files.readAllBytes(tmp)).isEqualTo(buffered);
    } finally {
      Files.deleteIfExists(tmp);
    }
  }

  @Test
  void streamedAndBufferedGenerationsAgreeByteForByte(@TempDir Path dir) throws IOException {
    int count = 6_000;
    SplittableRandom random = new SplittableRandom(7);
    List<Document> documents = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      documents.add(Document.of("v" + i, unitVector(random)));
    }

    // Appending with no quantizer streams vectors.bin; FLAT buffers it. The bytes must match.
    Path streamed = write(dir.resolve("streamed"), IndexType.HNSW, documents, 2_000);
    Path buffered = write(dir.resolve("buffered"), IndexType.FLAT, documents, 2_000);

    byte[] a = Files.readAllBytes(newestVectorsBin(streamed));
    byte[] b = Files.readAllBytes(newestVectorsBin(buffered));
    assertThat(a).as("streamed vectors.bin equals the buffered one").isEqualTo(b);
  }

  private static Path write(
      Path storage, IndexType indexType, List<Document> documents, int batch) {
    try (VectorCollection collection =
        VectorCollection.builder()
            .dimension(DIMENSION)
            .metric(SimilarityFunction.COSINE)
            .indexType(indexType)
            .storagePath(storage.toAbsolutePath())
            .build()) {
      for (int from = 0; from < documents.size(); from += batch) {
        collection.addAll(documents.subList(from, Math.min(from + batch, documents.size())));
        collection.commit();
      }
    }
    return storage;
  }

  private static Path newestVectorsBin(Path storage) throws IOException {
    try (var list = Files.list(storage)) {
      Path generation =
          list.filter(Files::isDirectory)
              .filter(p -> p.getFileName().toString().startsWith("gen-"))
              .max(java.util.Comparator.comparing(p -> p.getFileName().toString()))
              .orElseThrow();
      return generation.resolve("vectors.bin");
    }
  }

  private static float[] unitVector(SplittableRandom random) {
    float[] v = new float[DIMENSION];
    double sum = 0;
    for (int d = 0; d < DIMENSION; d++) {
      v[d] = (float) random.nextGaussian();
      sum += (double) v[d] * v[d];
    }
    float norm = (float) Math.sqrt(sum);
    for (int d = 0; d < DIMENSION; d++) {
      v[d] /= norm;
    }
    return v;
  }
}
