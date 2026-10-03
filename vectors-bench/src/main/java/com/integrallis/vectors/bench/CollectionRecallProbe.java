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

import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.storage.Manifest;
import com.integrallis.vectors.db.storage.MappedIdMapper;
import com.integrallis.vectors.db.storage.MemorySegmentVectors;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;

/** Exhaustive recall check for a replayed collection; source generation is mapped read-only. */
public final class CollectionRecallProbe {
  private static final int[] EFS = {10, 16, 32, 64, 128, 256};

  private CollectionRecallProbe() {}

  public static void main(String[] args) throws Exception {
    if (args.length != 4)
      throw new IllegalArgumentException("sourceGeneration replayedCollection limit queries");
    measure(
        Path.of(args[0]), Path.of(args[1]), Integer.parseInt(args[2]), Integer.parseInt(args[3]));
  }

  static double[] measure(Path source, Path target, int limit, int requestedQueries)
      throws Exception {
    if (limit < 1 || requestedQueries < 1)
      throw new IllegalArgumentException("positive limits required");
    if (!Files.isRegularFile(target.resolve("CURRENT")))
      throw new IllegalArgumentException("target must be an existing replayed collection");
    if (source.toRealPath().startsWith(target.toRealPath()))
      throw new IllegalArgumentException(
          "target must be separate from the original source collection");
    Manifest manifest = Manifest.readFrom(source.resolve("manifest.bin"));
    if (manifest.tombstoneCount() != 0 || manifest.vectorsBinLength() == 0)
      throw new IllegalArgumentException(
          "source must have full-precision vectors and no tombstones");
    int n = Math.toIntExact(manifest.liveCount());
    int count = Math.min(n, limit);
    if (count == 0) throw new IllegalArgumentException("source is empty");
    int qCount = Math.min(requestedQueries, count);
    int k = Math.min(10, count);
    try (Arena arena = Arena.ofConfined();
        var collection =
            VectorCollection.builder()
                .dimension(manifest.dimension())
                .metric(manifest.metric())
                .indexType(IndexType.HNSW)
                .storagePath(target)
                .build()) {
      if (collection.size() != count)
        throw new IllegalArgumentException("replay size does not match the source prefix");
      var vectors =
          MemorySegmentVectors.open(source.resolve("vectors.bin"), n, manifest.dimension(), arena);
      var ids = MappedIdMapper.open(source.resolve("idmap.bin"), arena);
      float[][] queries = new float[qCount][manifest.dimension()];
      String[][] truth = new String[qCount][k];
      long start = System.nanoTime();
      for (int q = 0; q < qCount; q++) {
        int ordinal = (int) ((long) q * count / qCount);
        MemorySegment query = vectors.vectorSlice(ordinal);
        MemorySegment.copy(query, ValueLayout.JAVA_FLOAT, 0, queries[q], 0, manifest.dimension());
        int[] best = new int[k];
        float[] scores = new float[k];
        Arrays.fill(scores, Float.NEGATIVE_INFINITY);
        for (int i = 0; i < count; i++) {
          float score =
              manifest.metric().compare(query, vectors.vectorSlice(i), manifest.dimension());
          if (!Float.isFinite(score))
            throw new IllegalArgumentException("nonfinite source similarity");
          if (score <= scores[k - 1]) continue;
          int position = k - 1;
          while (position > 0 && score > scores[position - 1]) {
            scores[position] = scores[position - 1];
            best[position] = best[position - 1];
            position--;
          }
          scores[position] = score;
          best[position] = i;
        }
        for (int j = 0; j < k; j++) truth[q][j] = ids.idOf(best[j]);
      }
      System.out.printf(
          Locale.ROOT,
          "ORACLE n=%d queries=%d k=%d sampling=evenly-spaced-source-rows exact_ms=%.3f%n",
          count,
          qCount,
          k,
          (System.nanoTime() - start) / 1e6);
      double[] recall = new double[EFS.length];
      System.out.println("ef,queries,recall10");
      for (int e = 0; e < EFS.length; e++) {
        int hits = 0;
        for (int q = 0; q < qCount; q++) {
          var result =
              collection.search(
                  SearchRequest.builder(queries[q], k).searchListSize(EFS[e]).build());
          for (var hit : result.hits())
            for (String correct : truth[q]) {
              if (correct.equals(hit.id())) {
                hits++;
                break;
              }
            }
        }
        recall[e] = hits / (double) (qCount * k);
        System.out.printf(Locale.ROOT, "%d,%d,%.6f%n", EFS[e], qCount, recall[e]);
      }
      return recall;
    }
  }
}
