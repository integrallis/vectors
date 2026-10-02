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

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.stream.Stream;

/**
 * What an ingest costs as a function of how often it commits.
 *
 * <p>A commit publishes a generation containing every live document plus the newly staged ones.
 * While each commit rebuilt the whole HNSW graph, an ingest split into K batches paid K builds of
 * growing size — quadratic in the number of commits, and the reason a large ingest appeared to be
 * embedding-bound when it was index-bound. This measures the wall time, the resulting recall, and
 * the bytes on disk for several cadences, so a regression shows up as the quadratic curve
 * returning.
 *
 * <p>Recall is reported alongside the time on purpose: a faster graph that retrieves worse is not
 * an improvement, and the cheapest way to make this benchmark look good would be to build a worse
 * graph.
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:run -PmainClass=com.integrallis.vectors.bench.CommitCadenceBenchmark \
 *     --args="100000 512"
 * }</pre>
 *
 * <p>Arguments: {@code [vectors] [dimension] [queries]}. Defaults: 50000 vectors, 512 dimensions,
 * 100 queries. Vectors are random unit vectors with a fixed seed — this measures build cost and
 * relative recall, not recall against a real corpus.
 */
public final class CommitCadenceBenchmark {

  private static final int K = 10;
  private static final long SEED = 1_234L;

  private CommitCadenceBenchmark() {}

  public static void main(String[] args) throws IOException {
    int count = args.length > 0 ? Integer.parseInt(args[0]) : 50_000;
    int dimension = args.length > 1 ? Integer.parseInt(args[1]) : 512;
    int queryCount = args.length > 2 ? Integer.parseInt(args[2]) : 100;
    IndexType indexType = args.length > 3 ? IndexType.valueOf(args[3]) : IndexType.HNSW;

    float[][] vectors = randomUnitVectors(count, dimension, SEED);
    float[][] queries = randomUnitVectors(queryCount, dimension, SEED + 1);
    List<Set<String>> truth = new ArrayList<>(queryCount);
    for (float[] query : queries) {
      truth.add(bruteForceTop(query, vectors));
    }

    System.out.printf(
        "%d vectors, %d dimensions, %s, cosine, %d cores%n",
        count, dimension, indexType, Runtime.getRuntime().availableProcessors());
    System.out.printf(
        "%n%-14s %10s %10s %8s %14s %8s%n",
        "commit every", "ms", "vec/s", "gens", "disk bytes", "recall");

    int[] cadences = {count, count / 2, count / 5, count / 10, count / 20, count / 40};
    for (int cadence : cadences) {
      if (cadence <= 0) {
        continue;
      }
      run(cadence, vectors, dimension, queries, truth, indexType);
    }
  }

  private static void run(
      int commitEvery,
      float[][] vectors,
      int dimension,
      float[][] queries,
      List<Set<String>> truth,
      IndexType indexType)
      throws IOException {

    Path dir = Files.createTempDirectory("commit-cadence-" + commitEvery);
    try {
      long start = System.nanoTime();
      double recall;
      long generations;
      try (VectorCollection collection =
          VectorCollection.builder()
              .dimension(dimension)
              .metric(SimilarityFunction.COSINE)
              .indexType(indexType)
              .storagePath(dir.toAbsolutePath())
              .build()) {

        List<Document> batch = new ArrayList<>(commitEvery);
        for (int i = 0; i < vectors.length; i++) {
          batch.add(Document.of("v" + i, vectors[i]));
          if (batch.size() == commitEvery) {
            collection.addAll(batch);
            collection.commit();
            batch.clear();
          }
        }
        if (!batch.isEmpty()) {
          collection.addAll(batch);
          collection.commit();
        }
        long millis = (System.nanoTime() - start) / 1_000_000;
        generations = collection.generationNumber();
        recall = recall(collection, queries, truth);

        long bytes = diskBytes(dir);
        System.out.printf(
            "%-14d %10d %10.0f %8d %14d %10.0f %8.4f%n",
            commitEvery,
            millis,
            vectors.length * 1000.0 / millis,
            generations,
            bytes,
            bytes / (double) vectors.length,
            recall);
      }
    } finally {
      deleteRecursively(dir);
    }
  }

  private static double recall(
      VectorCollection collection, float[][] queries, List<Set<String>> truth) {
    double total = 0;
    for (int q = 0; q < queries.length; q++) {
      SearchResult result = collection.search(SearchRequest.builder(queries[q], K).build());
      Set<String> found = new LinkedHashSet<>();
      for (SearchResult.Hit hit : result.hits()) {
        found.add(hit.id());
      }
      found.retainAll(truth.get(q));
      total += found.size() / (double) K;
    }
    return total / queries.length;
  }

  private static Set<String> bruteForceTop(float[] query, float[][] vectors) {
    Integer[] order = new Integer[vectors.length];
    for (int i = 0; i < vectors.length; i++) {
      order[i] = i;
    }
    Arrays.sort(
        order,
        (a, b) ->
            Float.compare(
                SimilarityFunction.COSINE.compare(query, vectors[b]),
                SimilarityFunction.COSINE.compare(query, vectors[a])));
    Set<String> top = new LinkedHashSet<>();
    for (int i = 0; i < K; i++) {
      top.add("v" + order[i]);
    }
    return top;
  }

  /**
   * Bytes actually occupied, counting each inode once.
   *
   * <p>A commit that only appends hard-links the predecessor's {@code vectors.bin} instead of
   * copying it, so the same inode appears in several generation directories. Summing file sizes per
   * directory would report it once per link and make an append look more expensive than a full
   * rewrite, which is the opposite of the truth. This is what {@code du} does.
   */
  private static long diskBytes(Path dir) throws IOException {
    Set<Object> seen = new java.util.HashSet<>();
    long total = 0;
    try (Stream<Path> walk = Files.walk(dir)) {
      for (Path path : (Iterable<Path>) walk.filter(Files::isRegularFile)::iterator) {
        Object key =
            Files.readAttributes(path, java.nio.file.attribute.BasicFileAttributes.class).fileKey();
        if (key != null && !seen.add(key)) {
          continue; // another link to a file already counted
        }
        total += Files.size(path);
      }
    }
    return total;
  }

  private static void deleteRecursively(Path root) throws IOException {
    if (!Files.exists(root)) {
      return;
    }
    try (Stream<Path> walk = Files.walk(root)) {
      walk.sorted(Comparator.reverseOrder())
          .forEach(
              p -> {
                try {
                  Files.delete(p);
                } catch (IOException ignored) {
                  // best effort on a temp directory
                }
              });
    }
  }

  private static float[][] randomUnitVectors(int count, int dimension, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    float[][] out = new float[count][dimension];
    for (int i = 0; i < count; i++) {
      double sum = 0;
      for (int d = 0; d < dimension; d++) {
        out[i][d] = (float) random.nextGaussian();
        sum += (double) out[i][d] * out[i][d];
      }
      float norm = (float) Math.sqrt(sum);
      for (int d = 0; d < dimension; d++) {
        out[i][d] /= norm;
      }
    }
    return out;
  }
}
