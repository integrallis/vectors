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
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Background-compaction generation reclamation (I.6): retired {@code gen-NNNN/} directories outside
 * the retention window are deleted from disk while the most recent ones are kept for crash
 * recovery.
 */
@Tag("unit")
class GenerationCompactionTest {

  private static final int DIM = 4;

  private static float[] vec(int i) {
    return new float[] {i, i + 1, i + 2, i + 3};
  }

  private static boolean genExists(Path root, long n) {
    return Files.isDirectory(root.resolve(FileFormat.generationDirName(n)));
  }

  @Test
  void pruneReclaimsOldGenerationsKeepingRetentionWindow(@TempDir Path tmp) {
    // Long interval so the daemon never fires mid-test; we drive a deterministic pass by hand.
    VectorCollectionImpl col =
        (VectorCollectionImpl)
            VectorCollection.builder()
                .dimension(DIM)
                .metric(SimilarityFunction.EUCLIDEAN)
                .indexType(IndexType.FLAT)
                .storagePath(tmp)
                .backgroundCompaction(Duration.ofHours(1))
                .retainGenerations(2)
                .build();
    try {
      // Bootstrap is gen-0; each commit lands the next generation. After 4 commits: gen-0..gen-4.
      for (int i = 1; i <= 4; i++) {
        col.add(Document.of("doc-" + i, vec(i)));
        col.commit();
      }
      // Retired generations linger on disk until reclaimed.
      assertThat(genExists(tmp, 0)).isTrue();
      assertThat(genExists(tmp, 1)).isTrue();
      assertThat(genExists(tmp, 4)).isTrue();

      col.pruneRetiredGenerations();

      // retain=2, current=4 -> keep gen-3 and gen-4; reclaim gen-0, gen-1, gen-2.
      assertThat(genExists(tmp, 0)).as("gen-0 reclaimed").isFalse();
      assertThat(genExists(tmp, 1)).as("gen-1 reclaimed").isFalse();
      assertThat(genExists(tmp, 2)).as("gen-2 reclaimed").isFalse();
      assertThat(genExists(tmp, 3)).as("gen-3 within retention window").isTrue();
      assertThat(genExists(tmp, 4)).as("current generation kept").isTrue();

      // The collection still serves correctly after reclamation.
      assertThat(col.size()).isEqualTo(4);
      SearchResult r = col.search(SearchRequest.builder(vec(4), 1).build());
      assertThat(r.hits().get(0).id()).isEqualTo("doc-4");
    } finally {
      col.close();
    }
  }

  @Test
  void daemonReclaimsAutomatically(@TempDir Path tmp) throws Exception {
    VectorCollectionImpl col =
        (VectorCollectionImpl)
            VectorCollection.builder()
                .dimension(DIM)
                .metric(SimilarityFunction.EUCLIDEAN)
                .indexType(IndexType.FLAT)
                .storagePath(tmp)
                .backgroundCompaction(Duration.ofMillis(50))
                .retainGenerations(2)
                .build();
    try {
      for (int i = 1; i <= 4; i++) {
        col.add(Document.of("doc-" + i, vec(i)));
        col.commit();
      }
      // The daemon should reclaim gen-0 within a few ticks.
      long deadline = System.nanoTime() + 10_000_000_000L;
      while (genExists(tmp, 0) && System.nanoTime() < deadline) {
        Thread.sleep(20);
      }
      assertThat(genExists(tmp, 0)).as("daemon reclaimed gen-0").isFalse();
      assertThat(genExists(tmp, 4)).as("current generation kept").isTrue();
    } finally {
      col.close();
    }
  }

  @Test
  void disabledByDefaultLeavesGenerationsOnDisk(@TempDir Path tmp) {
    try (VectorCollection col =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.FLAT)
            .storagePath(tmp)
            .build()) {
      for (int i = 1; i <= 3; i++) {
        col.add(Document.of("doc-" + i, vec(i)));
        col.commit();
      }
      // Without background compaction the historical generations remain (caller-driven cleanup).
      assertThat(genExists(tmp, 0)).isTrue();
      assertThat(genExists(tmp, 1)).isTrue();
      assertThat(genExists(tmp, 2)).isTrue();
      assertThat(genExists(tmp, 3)).isTrue();
    }
  }
}
