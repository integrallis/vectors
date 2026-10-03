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

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.hnsw.*;
import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.security.*;
import java.util.*;

/**
 * Exact construction timing and full graph digest; accepts an fbin corpus, count, threads, seed.
 */
public final class ExactConstructionBenchmark {
  public static void main(String[] args) throws Exception {
    Path source = Path.of(args[0]);
    int count = Integer.parseInt(args[1]);
    int threads = Integer.parseInt(args[2]);
    long seed = Long.parseLong(args[3]);
    boolean concurrent = args.length > 4 && args[4].equals("concurrent");
    float[][] data = read(source, count);
    int dimension = data[0].length;
    // Equal warmup before the measured build in each fresh JVM.
    build(Arrays.copyOf(data, Math.min(count, 5000)), threads, seed, concurrent);
    long start = System.nanoTime();
    var graph = build(data, threads, seed, concurrent);
    double seconds = (System.nanoTime() - start) / 1e9;
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    try (var out =
        new DataOutputStream(new DigestOutputStream(OutputStream.nullOutputStream(), digest))) {
      out.writeInt(graph.entryNode());
      out.writeInt(graph.maxLevel());
      out.writeInt(graph.size());
      for (int node = 0; node < graph.size(); node++) {
        out.writeInt(graph.nodeLevel(node));
        for (int level = 0; level <= graph.nodeLevel(node); level++) {
          var neighbors = graph.getNeighbors(node, level);
          out.writeInt(neighbors.size());
          for (int n = 0; n < neighbors.size(); n++) {
            out.writeInt(neighbors.node(n));
            out.writeInt(Float.floatToRawIntBits(neighbors.score(n)));
          }
        }
      }
    }
    System.out.printf(
        Locale.ROOT,
        "RESULT,%d,%d,%d,%d,%.9f,%s%n",
        count,
        dimension,
        threads,
        seed,
        seconds,
        HexFormat.of().formatHex(digest.digest()));
    if (args.length > 5) {
      float[][] queries = read(Path.of(args[5]), args.length > 6 ? Integer.parseInt(args[6]) : 100);
      var index =
          HnswIndex.ofPrebuilt(graph, new InMemoryVectors(data), SimilarityFunction.DOT_PRODUCT);
      int[] budgets = {10, 16, 32, 64, 128, 256};
      long[] hits = new long[budgets.length];
      for (float[] query : queries) {
        if (query.length != dimension)
          throw new IllegalArgumentException("query dimension differs");
        NodeQueue truth = new NodeQueue(10, true);
        for (int id = 0; id < count; id++)
          truth.insertWithOverflow(id, SimilarityFunction.DOT_PRODUCT.compare(query, data[id]), 10);
        Set<Integer> expected = new HashSet<>();
        while (!truth.isEmpty()) expected.add(NodeQueue.nodeId(truth.poll()));
        for (int b = 0; b < budgets.length; b++)
          for (int id : index.search(query, 10, budgets[b]).nodeIds())
            if (expected.contains(id)) hits[b]++;
      }
      for (int b = 0; b < budgets.length; b++)
        System.out.printf(
            Locale.ROOT,
            "RECALL,%d,%d,%.9f%n",
            budgets[b],
            queries.length,
            hits[b] / (double) (queries.length * 10));
    }
  }

  private static float[][] read(Path source, int count) throws IOException {
    float[][] data;
    int dimension;
    try (var in =
        new DataInputStream(new BufferedInputStream(Files.newInputStream(source), 1 << 20))) {
      int rows = Integer.reverseBytes(in.readInt());
      dimension = Integer.reverseBytes(in.readInt());
      if (count <= 0
          || count > rows
          || dimension <= 0
          || Files.size(source) != 8L + (long) rows * dimension * 4)
        throw new IllegalArgumentException("Invalid corpus shape or prefix");
      data = new float[count][dimension];
      byte[] bytes = new byte[dimension * 4];
      var row = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < count; i++) {
        in.readFully(bytes);
        row.rewind();
        for (int d = 0; d < dimension; d++) data[i][d] = row.getFloat();
      }
    }
    return data;
  }

  private static HnswGraph build(float[][] rows, int threads, long seed, boolean concurrent) {
    var vectors = new InMemoryVectors(rows);
    return concurrent || threads > 1
        ? ConcurrentHnswGraphBuilder.create(16, 200, vectors, SimilarityFunction.DOT_PRODUCT, seed)
            .build(threads)
        : HnswGraphBuilder.create(16, 200, vectors, SimilarityFunction.DOT_PRODUCT, seed).build();
  }
}
