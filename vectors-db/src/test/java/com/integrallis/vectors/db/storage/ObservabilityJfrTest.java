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
package com.integrallis.vectors.db.storage;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.hnsw.HnswGraph;
import com.integrallis.vectors.hnsw.HnswIndex;
import com.integrallis.vectors.quantization.ArrayVectorDataset;
import com.integrallis.vectors.quantization.ExtendedRaBitQuantizedVectors;
import com.integrallis.vectors.quantization.ExtendedRaBitQuantizer;
import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.storage.memory.AlignmentUtil;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class ObservabilityJfrTest {

  @Test
  void queryAndRangedGetJfrEventsFireWithCorrectData(@TempDir Path tmp) throws Exception {
    int n = 500;
    int dim = 64;
    int k = 10;
    float over = 3.0f;
    SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;
    float[][] vecs = random(n, dim, 1L);

    HnswGraph graph =
        HnswIndex.builder(vecs, sim).maxConnections(16).efConstruction(200).build().graph();
    ArrayVectorDataset ds = new ArrayVectorDataset(vecs);
    ExtendedRaBitQuantizedVectors codes = ExtendedRaBitQuantizer.train(ds, 4).encodeAll(ds);

    HeapStorageBackend backend = new HeapStorageBackend();
    backend.put("vectors.bin", encodeVectorsBin(vecs, dim));
    MappedHnswIndexAdapter adapter =
        new MappedHnswIndexAdapter(
            graph, new ObjectStoreRandomAccessVectors(backend, "vectors.bin", n, dim), sim);
    adapter.enableQuantization(codes);

    Path jfr = tmp.resolve("obs.jfr");
    try (Recording recording = new Recording()) {
      recording.enable("com.integrallis.vectors.Query");
      recording.enable("com.integrallis.vectors.RangedGet");
      recording.start();
      adapter.search(random(1, dim, 2L)[0], k, 100, over); // one search → one Query, ~30 RangedGets
      recording.stop();
      recording.dump(jfr);
    }

    List<RecordedEvent> events = RecordingFile.readAllEvents(jfr);
    List<RecordedEvent> queries =
        events.stream().filter(e -> e.getEventType().getName().endsWith(".Query")).toList();
    List<RecordedEvent> gets =
        events.stream().filter(e -> e.getEventType().getName().endsWith(".RangedGet")).toList();

    assertThat(queries).as("exactly one Query event").hasSize(1);
    RecordedEvent q = queries.get(0);
    assertThat(q.getInt("k")).isEqualTo(k);
    assertThat(q.getFloat("overQueryFactor")).isEqualTo(over);
    assertThat(q.getInt("results")).isEqualTo(k);
    assertThat(q.getDuration().toNanos()).as("Query event is timed").isPositive();

    // rerank fetches the over-query set: overQuery * k = 30 ranged GETs, each a RangedGet event.
    assertThat(gets).as("one RangedGet per rerank fetch").hasSize((int) (over * k));
    RecordedEvent g = gets.get(0);
    assertThat(g.getString("key")).isEqualTo("vectors.bin");
    assertThat(g.getInt("length")).isEqualTo(dim * Float.BYTES);
    assertThat(g.getLong("offset")).isGreaterThanOrEqualTo(0L);
  }

  @Test
  void eventsAreDisabledByDefaultWhenNoRecordingEnablesThem(@TempDir Path tmp) throws Exception {
    // With a recording that does NOT enable our events, shouldCommit() is false → no events
    // emitted.
    int n = 300;
    int dim = 32;
    float[][] vecs = random(n, dim, 5L);
    SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;
    HnswGraph graph =
        HnswIndex.builder(vecs, sim).maxConnections(16).efConstruction(200).build().graph();
    ArrayVectorDataset ds = new ArrayVectorDataset(vecs);
    ExtendedRaBitQuantizedVectors codes = ExtendedRaBitQuantizer.train(ds, 4).encodeAll(ds);
    HeapStorageBackend backend = new HeapStorageBackend();
    backend.put("vectors.bin", encodeVectorsBin(vecs, dim));
    MappedHnswIndexAdapter adapter =
        new MappedHnswIndexAdapter(
            graph, new ObjectStoreRandomAccessVectors(backend, "vectors.bin", n, dim), sim);
    adapter.enableQuantization(codes);

    Path jfr = tmp.resolve("none.jfr");
    try (Recording recording = new Recording()) {
      recording.start(); // no enable() for our events
      adapter.search(random(1, dim, 6L)[0], 10, 100, 3.0f);
      recording.stop();
      recording.dump(jfr);
    }
    long ours =
        RecordingFile.readAllEvents(jfr).stream()
            .filter(e -> e.getEventType().getName().startsWith("com.integrallis.vectors."))
            .count();
    assertThat(ours).as("disabled-by-default events emit nothing unless enabled").isZero();
  }

  private static byte[] encodeVectorsBin(float[][] vectors, int dim) {
    long stride = AlignmentUtil.alignUp((long) dim * Float.BYTES, AlignmentUtil.VECTOR_ALIGNMENT);
    byte[] blob = new byte[(int) (stride * vectors.length)];
    ByteBuffer buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < vectors.length; i++) {
      buf.position((int) (i * stride));
      for (int d = 0; d < dim; d++) {
        buf.putFloat(vectors[i][d]);
      }
    }
    return blob;
  }

  private static float[][] random(int n, int dim, long seed) {
    Random r = new Random(seed);
    float[][] v = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        v[i][d] = r.nextFloat() * 2f - 1f;
      }
    }
    return v;
  }
}
