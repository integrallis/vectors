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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Contract tests for collections that keep only their quantized codes.
 *
 * <p>The mode exists to make an index small enough to ship inside an artifact, so the size
 * assertion is the point rather than a detail: an ordinary quantizer <em>adds</em> a compressed
 * copy beside the full-precision vectors and makes the directory bigger.
 */
@Tag("unit")
class QuantizedOnlyCollectionTest {

  private static final int DIMENSION = 64;
  private static final int COUNT = 400;

  private static List<Document> documents(Random random) {
    return documents(random, "doc");
  }

  private static List<Document> documents(Random random, String prefix) {
    List<Document> documents = new ArrayList<>(COUNT);
    for (int index = 0; index < COUNT; index++) {
      float[] vector = new float[DIMENSION];
      double norm = 0;
      for (int component = 0; component < DIMENSION; component++) {
        vector[component] = (float) random.nextGaussian();
        norm += (double) vector[component] * vector[component];
      }
      float scale = (float) (1.0 / Math.sqrt(norm));
      for (int component = 0; component < DIMENSION; component++) {
        vector[component] *= scale;
      }
      documents.add(new Document(prefix + "-" + index, vector, null, null));
    }
    return documents;
  }

  private static VectorCollection open(Path storage, boolean quantizedOnly) {
    return VectorCollection.builder()
        .dimension(DIMENSION)
        .metric(SimilarityFunction.COSINE)
        .indexType(IndexType.FLAT)
        .quantizer(QuantizerKind.SQ4)
        .quantizedOnly(quantizedOnly)
        .storagePath(storage.toAbsolutePath())
        .build();
  }

  private static long directorySize(Path directory) throws IOException {
    try (var walk = Files.walk(directory)) {
      long total = 0;
      for (Path path : walk.filter(Files::isRegularFile).toList()) {
        total += Files.size(path);
      }
      return total;
    }
  }

  @Test
  void storesCodesInsteadOfVectorsAndIsSmallerForIt(@TempDir Path root) throws IOException {
    List<Document> documents = documents(new Random(7));
    Path quantizedOnly = root.resolve("quantized-only");
    Path full = root.resolve("full");

    try (VectorCollection collection = open(quantizedOnly, true)) {
      collection.addAll(documents);
      collection.commit();
    }
    try (VectorCollection collection = open(full, false)) {
      collection.addAll(documents);
      collection.commit();
    }

    long quantizedOnlyBytes = directorySize(quantizedOnly);
    long fullBytes = directorySize(full);
    assertThat(quantizedOnlyBytes).isLessThan(fullBytes);
    // Four bits per dimension against thirty-two, so the vectors themselves should shrink by
    // roughly eight. Asserted loosely because idmap and metadata do not compress with them.
    assertThat(quantizedOnlyBytes).isLessThan(fullBytes / 2);
  }

  @Test
  void findsTheSameNeighbourAsAnExactSearchForWellSeparatedVectors(@TempDir Path root) {
    List<Document> documents = documents(new Random(11));

    try (VectorCollection collection = open(root.resolve("index"), true)) {
      collection.addAll(documents);
      collection.commit();

      // Querying with a stored vector: approximate scores still have to rank its own row first,
      // which is the property a classifier over this index depends on.
      int agreed = 0;
      for (int index = 0; index < 50; index++) {
        SearchResult result =
            collection.search(SearchRequest.builder(documents.get(index).vector(), 1).build());
        assertThat(result.hits()).hasSize(1);
        if (result.hits().get(0).document().id().equals(documents.get(index).id())) {
          agreed++;
        }
      }
      assertThat(agreed).isEqualTo(50);
    }
  }

  @Test
  void reopensWithoutBeingToldItIsQuantizedOnly(@TempDir Path root) {
    List<Document> documents = documents(new Random(13));
    Path storage = root.resolve("index");
    try (VectorCollection collection = open(storage, true)) {
      collection.addAll(documents);
      collection.commit();
    }

    // Reopen with the ordinary defaults — neither a quantizer nor quantizedOnly is specified. The
    // manifest, not the caller's settings, decides how an existing generation is read. Otherwise
    // this reader would try to map a vectors.bin that is intentionally not there.
    try (VectorCollection reopened =
        VectorCollection.builder()
            .dimension(DIMENSION)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .storagePath(storage.toAbsolutePath())
            .build()) {
      SearchResult result =
          reopened.search(SearchRequest.builder(documents.get(0).vector(), 1).build());
      assertThat(result.hits()).hasSize(1);
      assertThat(result.hits().get(0).document().id()).isEqualTo(documents.get(0).id());
    }
  }

  @Test
  void isSealedAfterItsFirstCommit(@TempDir Path root) {
    List<Document> documents = documents(new Random(17));
    Path storage = root.resolve("index");
    try (VectorCollection collection = open(storage, true)) {
      collection.addAll(documents);
      collection.commit();
    }

    try (VectorCollection reopened = open(storage, true)) {
      reopened.addAll(documents(new Random(19), "later").subList(0, 5));
      // Re-encoding a successor would need the full-precision rows this collection discarded.
      // Refusing beats writing codes that silently omit everything already stored.
      assertThatThrownBy(reopened::commit)
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("sealed");
    }
  }

  @Test
  void refusesConfigurationsItCannotHonour(@TempDir Path root) {
    assertThatThrownBy(
            () ->
                VectorCollection.builder()
                    .dimension(DIMENSION)
                    .metric(SimilarityFunction.COSINE)
                    .indexType(IndexType.FLAT)
                    .quantizedOnly(true)
                    .storagePath(root.toAbsolutePath())
                    .build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("needs a quantizer");

    assertThatThrownBy(
            () ->
                VectorCollection.builder()
                    .dimension(DIMENSION)
                    .metric(SimilarityFunction.COSINE)
                    .indexType(IndexType.HNSW)
                    .quantizer(QuantizerKind.SQ4)
                    .quantizedOnly(true)
                    .storagePath(root.toAbsolutePath())
                    .build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("FLAT");

    assertThatThrownBy(
            () ->
                VectorCollection.builder()
                    .dimension(DIMENSION)
                    .metric(SimilarityFunction.COSINE)
                    .indexType(IndexType.FLAT)
                    .quantizer(QuantizerKind.SQ4)
                    .quantizedOnly(true)
                    .build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("persistent");
  }
}
