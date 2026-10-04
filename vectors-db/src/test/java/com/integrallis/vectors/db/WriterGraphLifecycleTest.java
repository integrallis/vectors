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

import static org.assertj.core.api.Assertions.*;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class WriterGraphLifecycleTest {
  @TempDir Path directory;

  private VectorCollection open(Path path, int threads) {
    var builder =
        VectorCollection.builder()
            .dimension(65)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.HNSW)
            .hnswBuildThreads(threads)
            .autoCommitThreshold(Integer.MAX_VALUE);
    if (path != null) builder.storagePath(path);
    return builder.build();
  }

  private static float[][] data() {
    Random random = new Random(123);
    float[][] rows = new float[800][65];
    for (float[] row : rows) for (int d = 0; d < row.length; d++) row[d] = random.nextFloat() - .5f;
    return rows;
  }

  private static void add(VectorCollection collection, float[][] rows, int from, int to) {
    for (int i = from; i < to; i++) collection.add(Document.of("v" + i, rows[i]));
    collection.commit();
  }

  private static void check(VectorCollection collection, float[][] rows, int from, int to) {
    for (int i = from; i < to; i++) {
      assertThat(collection.get("v" + i)).isNotNull();
      assertThat(collection.get("v" + i).vector()).containsExactly(rows[i]);
    }
    for (int i = from; i < to; i += 7) {
      // A concurrent HNSW graph can contain unreachable nodes even with a full-size beam.
      // Check storage/result alignment here; fixed-budget benchmarks separately gate recall.
      var hits =
          collection
              .search(SearchRequest.builder(rows[i], 1).searchListSize(rows.length).build())
              .hits();
      assertThat(hits).isNotEmpty();
      var hit = hits.getFirst();
      int returned = Integer.parseInt(hit.id().substring(1));
      assertThat(hit.document().vector()).containsExactly(rows[returned]);
      assertThat(hit.score())
          .isCloseTo(SimilarityFunction.COSINE.compare(rows[i], rows[returned]), within(1e-6f));
    }
  }

  private static List<String> searchIds(VectorCollection collection, float[][] rows, int count) {
    List<String> ids = new ArrayList<>();
    for (int i = 0; i < count; i += 7) {
      ids.add(
          collection
              .search(SearchRequest.builder(rows[i], 1).searchListSize(rows.length).build())
              .hits()
              .getFirst()
              .id());
    }
    return ids;
  }

  @Test
  void compactingAnEmptyPersistentCollectionReleasesGraphAndLocksAndAllowsFreshIngest()
      throws Exception {
    float[][] rows = data();
    try (var c = open(directory, 2)) {
      add(c, rows, 0, 200);
      for (int i = 0; i < 200; i++) c.delete("v" + i);
      c.commit();
      var cache = c.getClass().getDeclaredField("writerGraph");
      cache.setAccessible(true);
      assertThat(cache.get(c)).isNotNull();
      c.compact();
      assertThat(c.size()).isZero();
      assertThat(cache.get(c)).isNull();
      assertThat(field(c, "writerLocks")).isNull();
      add(c, rows, 200, 300);
      check(c, rows, 200, 300);
    }
  }

  @Test
  void upsertAndDeleteBeforeAppendKeepVectorsAlignedWithPhysicalOrdinals() {
    float[][] rows = data();
    for (Path path : new Path[] {null, directory.resolve("persistent-upsert")}) {
      try (var c = open(path, 2)) {
        add(c, rows, 0, 500);
        c.upsert(Document.of("v17", rows[799]));
        c.delete("v10");
        c.commit();
        add(c, rows, 500, 600);
        assertThat(c.size()).isEqualTo(599);
        assertThat(c.get("v10")).isNull();
        assertThat(c.get("v17").vector()).containsExactly(rows[799]);
        assertThat(
                c.search(SearchRequest.builder(rows[799], 1).searchListSize(800).build())
                    .hits()
                    .getFirst()
                    .id())
            .isEqualTo("v17");
        check(c, rows, 500, 600);
      }
    }
  }

  @Test
  void closeReleasesTheWriterGraphAndLocks() throws Exception {
    var collection = open(directory, 1);
    add(collection, data(), 0, 200);
    var cache = collection.getClass().getDeclaredField("writerGraph");
    cache.setAccessible(true);
    assertThat(cache.get(collection)).isNotNull();
    collection.close();
    assertThat(cache.get(collection)).isNull();
    assertThat(field(collection, "writerLocks")).isNull();
  }

  @RepeatedTest(10)
  void failedPublicationDiscardsTheMutatedCacheAndRetryKeepsAllRows() throws Exception {
    float[][] rows = data();
    List<String> committedResults;
    try (var c = open(directory, 2)) {
      add(c, rows, 0, 200);
      add(c, rows, 200, 350); // creates append headroom
      long published = c.generationNumber();
      Path graph = directory.resolve("gen-%016d".formatted(published)).resolve("graph.bin");
      byte[] publishedGraph = Files.readAllBytes(graph);
      Path blocker = directory.resolve(".gen-%016d.tmp".formatted(published + 1));
      Files.createDirectory(blocker); // writeGeneration refuses a pre-existing in-flight directory
      for (int i = 350; i < 500; i++) c.add(Document.of("v" + i, rows[i]));
      assertThatThrownBy(c::commit).isInstanceOf(UncheckedIOException.class);
      assertThat(c.generationNumber()).isEqualTo(published);
      assertThat(Files.readAllBytes(graph)).isEqualTo(publishedGraph);
      var cache = c.getClass().getDeclaredField("writerGraph");
      cache.setAccessible(true);
      assertThat(cache.get(c)).isNull();
      assertThat(field(c, "writerLocks")).isNull();
      Files.delete(blocker);
      c.commit();
      assertThat(c.size()).isEqualTo(500);
      check(c, rows, 0, 500);
      committedResults = searchIds(c, rows, 500);
    }
    try (var c = open(directory, 1)) {
      assertThat(c.size()).isEqualTo(500);
      check(c, rows, 0, 500);
      assertThat(searchIds(c, rows, 500)).containsExactlyElementsOf(committedResults);
    }
  }

  @Test
  void persistentAppendReopenAndCompaction() {
    float[][] rows = data();
    try (var c = open(directory, 2)) {
      add(c, rows, 0, 200);
      add(c, rows, 200, 350);
      add(c, rows, 350, 500); // exercises in-place append and graph reuse
      assertThat(c.config().quantizerKind()).isEqualTo(QuantizerKind.NONE);
      check(c, rows, 0, 500);
    }
    try (var c = open(directory, 2)) {
      add(c, rows, 500, 600); // decoded graph has synthetic scores; recompute from stored vectors
      c.delete("v0");
      c.commit();
      c.compact(); // ordinals change; invalidate cached graph
      add(c, rows, 600, 700);
      check(c, rows, 1, 700);
    }
    try (var c = open(directory, 1)) {
      add(c, rows, 700, 800); // reopen must restore exact edge scores
      assertThat(c.size()).isEqualTo(799);
      check(c, rows, 1, 800);
    }
  }

  @Test
  void inMemoryAppendAfterCompaction() {
    float[][] rows = data();
    try (var c = open(null, 1)) {
      add(c, rows, 0, 300);
      add(c, rows, 300, 500);
      c.delete("v0");
      c.commit();
      c.compact();
      add(c, rows, 500, 800);
      assertThat(c.size()).isEqualTo(799);
      check(c, rows, 1, 800);
    }
  }

  private static Object field(VectorCollection collection, String name) throws Exception {
    var field = collection.getClass().getDeclaredField(name);
    field.setAccessible(true);
    return field.get(collection);
  }

  @Test
  void successfulAppendReusesGraphLocksAndExecutor() throws Exception {
    float[][] rows = data();
    try (var c = open(directory, 2)) {
      add(c, rows, 0, 200);
      add(c, rows, 200, 350);
      add(c, rows, 350, 400);
      Object graph = field(c, "writerGraph");
      Object locks = field(c, "writerLocks");
      Object executor = field(c, "writerGraphExecutor");
      assertThat(executor).isNotNull();
      add(c, rows, 400, 500);
      assertThat(field(c, "writerGraph")).isSameAs(graph);
      assertThat(field(c, "writerLocks")).isSameAs(locks);
      assertThat(field(c, "writerGraphExecutor")).isSameAs(executor);
      check(c, rows, 0, 500);
    }
  }

  @Test
  void refreshReleasesGraphAndLocksBeforeTheNextAppend() throws Exception {
    float[][] rows = data();
    Path target = directory.resolve("target");
    Path remote = directory.resolve("remote");
    try (var c = open(target, 2)) {
      add(c, rows, 0, 200);
      add(c, rows, 200, 350);
      copyTree(target, remote);
      long nextGeneration;
      try (var other = open(remote, 2)) {
        add(other, rows, 350, 500);
        nextGeneration = other.generationNumber();
      }
      String generation = "gen-%016d".formatted(nextGeneration);
      copyTree(remote.resolve(generation), target.resolve(generation));
      Files.copy(
          remote.resolve("CURRENT"),
          target.resolve("CURRENT"),
          java.nio.file.StandardCopyOption.REPLACE_EXISTING);
      assertThat(c.refresh()).isTrue();
      assertThat(field(c, "writerGraph")).isNull();
      assertThat(field(c, "writerLocks")).isNull();
      add(c, rows, 500, 600);
      check(c, rows, 0, 600);
    }
  }

  private static void copyTree(Path from, Path to) throws Exception {
    try (var paths = Files.walk(from)) {
      for (Path source : paths.toList()) {
        Path target = to.resolve(from.relativize(source));
        if (Files.isDirectory(source)) Files.createDirectories(target);
        else Files.copy(source, target);
      }
    }
  }
}
