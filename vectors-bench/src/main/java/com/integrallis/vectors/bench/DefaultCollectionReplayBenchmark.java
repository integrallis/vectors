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
import com.integrallis.vectors.db.*;
import com.integrallis.vectors.db.storage.*;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/** Replays a pinned, read-only generation into a NEW collection, without invoking an embedder. */
public final class DefaultCollectionReplayBenchmark {
  private DefaultCollectionReplayBenchmark() {}

  public record Result(
      int documents, int commits, long elapsedNanos, long reopenNanos, int efConstruction) {}

  public static void main(String[] args) throws Exception {
    if (args.length != 5)
      throw new IllegalArgumentException("sourceGeneration destination cadence threads limit");
    System.out.println("arm,documents,commits,ingest_ms,reopen_ms");
    var result =
        run(
            Path.of(args[0]),
            Path.of(args[1]),
            Integer.parseInt(args[2]),
            Integer.parseInt(args[3]),
            Integer.parseInt(args[4]));
    System.out.printf(
        "META efConstruction=%d initialLevelSeed=time-generated%n", result.efConstruction());
    System.out.printf(
        Locale.ROOT,
        "%s,%d,%d,%.3f,%.3f%n",
        "default",
        result.documents(),
        result.commits(),
        result.elapsedNanos() / 1e6,
        result.reopenNanos() / 1e6);
  }

  static Result run(Path source, Path destination, int cadence, int threads, int limit)
      throws Exception {
    if (Files.exists(destination))
      throw new IllegalArgumentException("destination must not exist: " + destination);
    if (cadence < 1 || threads < 1 || limit < 1)
      throw new IllegalArgumentException("positive cadence, threads and limit required");
    Manifest manifest = Manifest.readFrom(source.resolve("manifest.bin"));
    if (manifest.tombstoneCount() != 0 || manifest.vectorsBinLength() == 0)
      throw new IllegalArgumentException(
          "replay requires a full-precision generation without tombstones");
    int n = Math.toIntExact(manifest.liveCount());
    int count = Math.min(n, limit);
    int commits = 0;
    long elapsed;
    int effectiveEfConstruction;
    try (Arena arena = Arena.ofConfined()) {
      var vectors =
          MemorySegmentVectors.open(source.resolve("vectors.bin"), n, manifest.dimension(), arena);
      var ids = MappedIdMapper.open(source.resolve("idmap.bin"), arena);
      var metadata = MappedMetadataStore.open(source.resolve("metadata.bin"), arena);
      if (ids.size() != n || metadata.size() != n)
        throw new IllegalStateException("generation row counts disagree");
      long start = System.nanoTime();
      try (var target = builder(destination, manifest, threads).build()) {
        effectiveEfConstruction = target.config().hnswParams().efConstruction();
        for (int i = 0; i < count; i++) {
          float[] vector = new float[manifest.dimension()];
          MemorySegment.copy(
              vectors.vectorSlice(i), ValueLayout.JAVA_FLOAT, 0, vector, 0, vector.length);
          Document doc = metadata.get(i);
          target.add(
              new Document(ids.idOf(i), vector, doc.text(), doc.metadata(), doc.contentHash()));
          if ((i + 1) % cadence == 0 || i + 1 == count) {
            long before = System.nanoTime();
            target.commit();
            commits++;
            System.out.printf(
                Locale.ROOT, "COMMIT rows=%d ms=%.3f%n", i + 1, (System.nanoTime() - before) / 1e6);
          }
        }
        if (target.size() != count) throw new IllegalStateException("replay lost documents");
      }
      elapsed = System.nanoTime() - start;
    }
    long start = System.nanoTime();
    try (var reopened = builder(destination, manifest, threads).build()) {
      if (reopened.size() != count) throw new IllegalStateException("reopen lost documents");
    }
    return new Result(count, commits, elapsed, System.nanoTime() - start, effectiveEfConstruction);
  }

  private static VectorCollectionBuilder builder(Path destination, Manifest source, int threads) {
    return VectorCollection.builder()
        .dimension(source.dimension())
        .metric(source.metric())
        .normalizeCosineVectors(source.vectorsNormalized())
        .indexType(IndexType.HNSW)
        .hnswBuildThreads(threads)
        .autoCommitThreshold(Integer.MAX_VALUE)
        .storagePath(destination.toAbsolutePath());
  }
}
