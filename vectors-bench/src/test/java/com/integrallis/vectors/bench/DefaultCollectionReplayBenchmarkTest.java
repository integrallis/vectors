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
package com.integrallis.vectors.bench;

import static org.assertj.core.api.Assertions.*;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.*;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class DefaultCollectionReplayBenchmarkTest {
  @TempDir Path root;

  @Test
  void preservesNormalizedCosineConfiguration() throws Exception {
    Path source = root.resolve("normalized-source");
    long generation;
    try (var c =
        VectorCollection.builder()
            .dimension(3)
            .metric(SimilarityFunction.COSINE)
            .normalizeCosineVectors(true)
            .indexType(IndexType.HNSW)
            .storagePath(source)
            .build()) {
      for (int i = 1; i <= 30; i++) c.add(Document.of("id" + i, new float[] {i, 1, 2}));
      c.commit();
      generation = c.generationNumber();
    }
    Path destination = root.resolve("normalized-replay");
    DefaultCollectionReplayBenchmark.run(
        source.resolve("gen-%016d".formatted(generation)), destination, 17, 1, 30);
    try (var c =
        VectorCollection.builder()
            .dimension(3)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.HNSW)
            .storagePath(destination)
            .build()) {
      var manifest =
          com.integrallis.vectors.db.storage.Manifest.readFrom(
              destination
                  .resolve("gen-%016d".formatted(c.generationNumber()))
                  .resolve("manifest.bin"));
      assertThat(manifest.vectorsNormalized()).isTrue();
    }
  }

  @Test
  void replaysCachedVectorsAndTextWithoutModifyingSource() throws Exception {
    Path source = root.resolve("source");
    long generation;
    try (var collection =
        VectorCollection.builder()
            .dimension(3)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.HNSW)
            .storagePath(source)
            .build()) {
      for (int i = 0; i < 50; i++)
        collection.add(Document.of("id" + i, new float[] {i, i + 1, i + 2}, "text" + i));
      collection.commit();
      generation = collection.generationNumber();
    }
    Path gen = source.resolve("gen-%016d".formatted(generation));
    byte[] before = Files.readAllBytes(gen.resolve("vectors.bin"));
    Path output = root.resolve("replay");
    var result = DefaultCollectionReplayBenchmark.run(gen, output, 17, 2, 50);
    assertThat(result.documents()).isEqualTo(50);
    assertThat(result.commits()).isEqualTo(3);
    assertThat(result.efConstruction()).isEqualTo(200);
    double[] recall = CollectionRecallProbe.measure(gen, output, 50, 10);
    assertThat(recall).hasSize(6);
    assertThat(recall[5]).isEqualTo(1.0);
    assertThatThrownBy(() -> CollectionRecallProbe.measure(gen, output, 40, 10))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> CollectionRecallProbe.measure(gen, source, 50, 10))
        .isInstanceOf(IllegalArgumentException.class);
    Path missingTarget = root.resolve("missing-replay");
    assertThatThrownBy(() -> CollectionRecallProbe.measure(gen, missingTarget, 50, 10))
        .isInstanceOf(IllegalArgumentException.class);
    assertThat(missingTarget).doesNotExist();
    assertThat(Files.readAllBytes(gen.resolve("vectors.bin"))).isEqualTo(before);
    try (var collection =
        VectorCollection.builder()
            .dimension(3)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.HNSW)
            .storagePath(output)
            .build()) {
      assertThat(collection.size()).isEqualTo(50);
      assertThat(collection.get("id23").text()).isEqualTo("text23");
      assertThat(collection.get("id23").vector()).containsExactly(23f, 24f, 25f);
    }
    assertThatThrownBy(() -> DefaultCollectionReplayBenchmark.run(gen, output, 17, 2, 50))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
