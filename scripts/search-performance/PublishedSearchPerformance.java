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
import com.integrallis.vectors.core.*;
import com.integrallis.vectors.db.index.*;
import com.integrallis.vectors.db.storage.*;
import com.integrallis.vectors.hnsw.*;
import com.integrallis.vectors.storage.store.VectorStoreWriter;
import java.lang.foreign.Arena;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

/** Published corpus prefixes and held-out queries; graph and exact oracle frozen by baseline. */
public final class PublishedSearchPerformance {
  static float[][] read(Path file) throws Exception {
    var bytes = ByteBuffer.wrap(Files.readAllBytes(file)).order(ByteOrder.LITTLE_ENDIAN);
    int n = bytes.getInt(), d = bytes.getInt();
    float[][] rows = new float[n][d];
    for (float[] row : rows) for (int i = 0; i < d; i++) row[i] = bytes.getFloat();
    if (bytes.hasRemaining()) throw new IllegalArgumentException("trailing input");
    return rows;
  }

  public static void main(String[] args) throws Exception {
    Path input = Path.of(args[0]), output = Path.of(args[1]);
    Files.createDirectories(output);
    for (String name : new String[] {"glove-100-angular", "fashion-mnist-784-euclidean"}) {
      var metric =
          name.startsWith("glove") ? SimilarityFunction.COSINE : SimilarityFunction.EUCLIDEAN;
      float[][] data = read(input.resolve(name + "-train.fbin"));
      float[][] queries = read(input.resolve(name + "-test.fbin"));
      int dim = data[0].length;
      var flat = new FlatScanAdapter();
      flat.build(data, metric);
      // Flat timings use the first 32 official test queries; HNSW uses all 256.
      float[][] flatQueries = Arrays.copyOf(queries, 32);
      CollectionSearchPerformance.flat("flat-" + name + "-heap", flat, flatQueries, false, dim);
      CollectionSearchPerformance.flat("flat-" + name + "-batch", flat, flatQueries, true, dim);
      Path vectors = output.resolve(name + "-vectors.bin");
      if (!Files.exists(vectors)) {
        try (var writer = VectorStoreWriter.open(vectors, dim, VectorEncoding.FLOAT32)) {
          for (var row : data) writer.writeVector(row);
        }
      }
      try (var arena = Arena.ofConfined()) {
        var store = MemorySegmentVectors.open(vectors, data.length, dim, arena);
        CollectionSearchPerformance.flat(
            "flat-" + name + "-mapped",
            new MappedFlatScanAdapter(store, metric),
            flatQueries,
            false,
            dim);
      }
      float[][] graphData = Arrays.copyOf(data, 20000);
      var source = new InMemoryVectors(graphData);
      Path savedGraph = output.resolve(name + "-graph.bin");
      if (!Files.exists(savedGraph)) {
        var graph = HnswGraphBuilder.create(16, 200, source, metric, 421).build();
        Files.write(savedGraph, HnswGraphCodec.encode(graph));
      }
      var index =
          HnswIndex.ofPrebuilt(
              HnswGraphCodec.decode(Files.readAllBytes(savedGraph)), source, metric);
      Path oracleFile = output.resolve(name + "-oracle.bin");
      if (!Files.exists(oracleFile)) {
        flat.build(graphData, metric);
        var oracle = ByteBuffer.allocate(queries.length * 10 * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (var q : queries)
          for (int id : flat.search(q, 10, 128, 1f).ordinals()) oracle.putInt(id);
        Files.write(oracleFile, oracle.array());
      }
      var oracle = ByteBuffer.wrap(Files.readAllBytes(oracleFile)).order(ByteOrder.LITTLE_ENDIAN);
      int[][] expected = new int[queries.length][10];
      for (int[] row : expected) for (int i = 0; i < 10; i++) row[i] = oracle.getInt();
      for (int ef : new int[] {32, 128, 512}) {
        for (int warm = 0; warm < 12; warm++)
          SearchPerformance.graphPass(index, queries, ef, false);
        long hash = SearchPerformance.graphPass(index, queries, ef, false);
        int hits = 0;
        for (int q = 0; q < queries.length; q++) {
          for (int id : index.search(queries[q], 10, ef).nodeIds()) {
            for (int truth : expected[q])
              if (id == truth) {
                hits++;
                break;
              }
          }
        }
        System.out.printf(
            Locale.ROOT,
            "RECALL,%s,%d,%d,%d,%.8f%n",
            name,
            ef,
            hits,
            queries.length * 10,
            hits / (queries.length * 10.0));
        for (int round = 0; round < 7; round++) {
          long start = System.nanoTime();
          long got = SearchPerformance.graphPass(index, queries, ef, false);
          long ns = System.nanoTime() - start;
          if (hash != got) throw new AssertionError("unstable results");
          System.out.printf(
              Locale.ROOT, "hnsw-%s-%d,%d,20000,10,%d,%d,%d%n", name, ef, dim, round, ns, hash);
        }
      }
    }
  }
}
