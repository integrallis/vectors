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
import com.integrallis.vectors.db.storage.FileFormat;
import com.integrallis.vectors.db.storage.Manifest;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * A commit that only adds documents must write the batch, not the collection.
 *
 * <p>{@code vectors.bin} is a bare array of stride-aligned records, so a successor's bytes are the
 * predecessor's followed by the new ones. The commit hard-links the predecessor's file and appends,
 * which makes two things observable: the two generations share an inode, and the bytes are
 * identical to what a full rewrite would have produced.
 */
class CommitAppendsVectorsFileTest {

  private static final int DIMENSION = 128;

  @Test
  void successiveGenerationsShareTheVectorsFileAndItsContentIsCorrect(@TempDir Path dir)
      throws IOException {
    Path storage = dir.resolve("linked");
    List<Document> all = documents(6_000, 11L);

    try (VectorCollection collection = open(storage, IndexType.HNSW)) {
      for (int from = 0; from < all.size(); from += 2_000) {
        collection.addAll(all.subList(from, from + 2_000));
        collection.commit();
      }
      assertThat(collection.size()).isEqualTo(6_000);
    }

    List<Path> generations = generations(storage);
    assertThat(generations).hasSizeGreaterThanOrEqualTo(3);

    Path last = generations.getLast();
    Path previous = generations.get(generations.size() - 2);

    // Shared inode: the append reused the predecessor's file instead of copying it.
    Object lastKey =
        Files.readAttributes(last.resolve(FileFormat.VECTORS_FILE), "unix:ino").get("ino");
    Object previousKey =
        Files.readAttributes(previous.resolve(FileFormat.VECTORS_FILE), "unix:ino").get("ino");
    assertThat(lastKey).as("vectors.bin inode shared between generations").isEqualTo(previousKey);

    // The predecessor's manifest still describes a shorter prefix of that shared file, and the file
    // on disk is longer. That is the invariant the whole design rests on.
    Manifest previousManifest = Manifest.readFrom(previous.resolve(FileFormat.MANIFEST_FILE));
    Manifest lastManifest = Manifest.readFrom(last.resolve(FileFormat.MANIFEST_FILE));
    assertThat(previousManifest.vectorsBinLength()).isLessThan(lastManifest.vectorsBinLength());
    assertThat(Files.size(previous.resolve(FileFormat.VECTORS_FILE)))
        .isEqualTo(lastManifest.vectorsBinLength());

    // And the bytes equal what a collection built with a single commit produces.
    Path reference = dir.resolve("reference");
    try (VectorCollection collection = open(reference, IndexType.HNSW)) {
      collection.addAll(all);
      collection.commit();
    }
    byte[] linked = prefix(last.resolve(FileFormat.VECTORS_FILE), lastManifest.vectorsBinLength());
    Path referenceGen = generations(reference).getLast();
    Manifest referenceManifest = Manifest.readFrom(referenceGen.resolve(FileFormat.MANIFEST_FILE));
    byte[] whole =
        prefix(referenceGen.resolve(FileFormat.VECTORS_FILE), referenceManifest.vectorsBinLength());
    assertThat(linked).as("appended bytes equal a full rewrite").isEqualTo(whole);
    assertThat(lastManifest.vectorsBinCrc32()).isEqualTo(referenceManifest.vectorsBinCrc32());
  }

  @Test
  void reopeningAfterLinkedCommitsReadsEveryDocument(@TempDir Path dir) {
    Path storage = dir.resolve("reopen");
    List<Document> all = documents(5_000, 21L);
    try (VectorCollection collection = open(storage, IndexType.HNSW)) {
      for (int from = 0; from < all.size(); from += 1_000) {
        collection.addAll(all.subList(from, from + 1_000));
        collection.commit();
      }
    }
    try (VectorCollection reopened = open(storage, IndexType.HNSW)) {
      assertThat(reopened.size()).isEqualTo(5_000);
      for (int i = 0; i < 5_000; i += 500) {
        assertThat(reopened.get("v" + i)).as("document v%d survives", i).isNotNull();
      }
      SearchResult result =
          reopened.search(SearchRequest.builder(all.get(1_234).vector(), 1).build());
      assertThat(result.hits().getFirst().id()).isEqualTo("v1234");
    }
  }

  /**
   * A commit that dies after appending leaves a tail in the predecessor's file. The predecessor
   * must still open, because its extent is its manifest's length and not the file's, and the next
   * commit must overwrite that tail rather than build on it.
   */
  @Test
  void aStaleTailFromAFailedCommitIsHarmlessAndOverwritten(@TempDir Path dir) throws IOException {
    Path storage = dir.resolve("tail");
    List<Document> first = documents(2_000, 31L);
    try (VectorCollection collection = open(storage, IndexType.HNSW)) {
      collection.addAll(first);
      collection.commit();
    }

    Path generation = generations(storage).getLast();
    Manifest manifest = Manifest.readFrom(generation.resolve(FileFormat.MANIFEST_FILE));
    Path vectors = generation.resolve(FileFormat.VECTORS_FILE);

    // Simulate a commit that appended and then died: garbage past the declared length.
    try (FileChannel channel = FileChannel.open(vectors, StandardOpenOption.WRITE)) {
      channel.position(manifest.vectorsBinLength());
      channel.write(java.nio.ByteBuffer.wrap(new byte[4096]));
      channel.force(true);
    }
    assertThat(Files.size(vectors)).isGreaterThan(manifest.vectorsBinLength());

    // The generation still opens and still serves its documents.
    try (VectorCollection reopened = open(storage, IndexType.HNSW)) {
      assertThat(reopened.size()).isEqualTo(2_000);

      // And the next commit discards the tail rather than appending after it.
      reopened.addAll(documents(500, 41L, 2_000));
      reopened.commit();
      assertThat(reopened.size()).isEqualTo(2_500);
    }
    Path newest = generations(storage).getLast();
    Manifest after = Manifest.readFrom(newest.resolve(FileFormat.MANIFEST_FILE));
    assertThat(Files.size(newest.resolve(FileFormat.VECTORS_FILE)))
        .as("no stale tail survives the next commit")
        .isEqualTo(after.vectorsBinLength());
  }

  // -------------------------------------------------------------------------

  private static VectorCollection open(Path storage, IndexType indexType) {
    return VectorCollection.builder()
        .dimension(DIMENSION)
        .metric(SimilarityFunction.COSINE)
        .indexType(indexType)
        .storagePath(storage.toAbsolutePath())
        .build();
  }

  private static List<Path> generations(Path storage) throws IOException {
    try (var list = Files.list(storage)) {
      return list.filter(Files::isDirectory)
          .filter(p -> p.getFileName().toString().startsWith("gen-"))
          .filter(p -> Files.exists(p.resolve(FileFormat.MANIFEST_FILE)))
          .sorted(Comparator.comparing(p -> p.getFileName().toString()))
          .toList();
    }
  }

  private static byte[] prefix(Path file, long length) throws IOException {
    byte[] out = new byte[(int) length];
    try (var in = Files.newInputStream(file)) {
      int read = in.readNBytes(out, 0, out.length);
      assertThat(read).isEqualTo(out.length);
    }
    return out;
  }

  private static List<Document> documents(int count, long seed) {
    return documents(count, seed, 0);
  }

  private static List<Document> documents(int count, long seed, int idOffset) {
    SplittableRandom random = new SplittableRandom(seed);
    List<Document> out = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
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
      out.add(Document.of("v" + (idOffset + i), v));
    }
    return out;
  }
}
