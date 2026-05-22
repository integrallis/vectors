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
import com.integrallis.vectors.db.id.InMemoryIdMapper;
import com.integrallis.vectors.db.metadata.InMemoryMetadataStore;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * JMH benchmark for the commit-time cost of {@link InMemoryIdMapper#copyOf} + {@link
 * InMemoryMetadataStore#copyOf} and the per-snapshot read cost (forward / reverse lookups).
 *
 * <p>These two structures dominate the per-commit allocation in {@code
 * VectorCollectionImpl.commitInMemory} now that the float-vector matrix is shared by reference. The
 * bench measures (1) {@code copyOf} on the predecessor, (2) bulk add of a staged batch on the
 * successor, (3) bulk read against the resulting successor. Output is per-commit cost at realistic
 * collection sizes; we use it to size the speedup target for a persistent-collection (HAMT)
 * replacement.
 *
 * <p>Run:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=CommitPipelineBenchmark
 * }</pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class CommitPipelineBenchmark {

  /** Live collection size to simulate (number of entries already in the predecessor). */
  @Param({"1000", "10000", "100000"})
  int liveSize;

  /** Staged batch size to add to the successor (typical user commit). */
  @Param({"100", "1000"})
  int batchSize;

  /** Number of reads to perform against the successor (simulates a top-K materialization). */
  private static final int READ_COUNT = 10;

  /** Pre-built predecessor state, set up once per trial. */
  private InMemoryIdMapper predecessorIdMapper;

  private InMemoryMetadataStore predecessorMetaStore;

  /** Pre-built staged batch payload, set up once per trial. */
  private String[] stagedIds;

  private Document[] stagedDocs;

  /** Pre-built read workload (a fixed set of ordinals to fetch). */
  private int[] readOrdinals;

  private String[] readIds;

  @Setup(Level.Trial)
  public void setup() {
    Random rng = new Random(42L);

    // Build the predecessor: liveSize sequential ids, each with a Document containing a 128-dim
    // vector. Same shape as VectorDbConcurrencyTest's predecessor at end-of-test.
    predecessorIdMapper = new InMemoryIdMapper();
    predecessorMetaStore = new InMemoryMetadataStore();
    for (int i = 0; i < liveSize; i++) {
      String id = "doc-" + i;
      int ord = predecessorIdMapper.put(id);
      float[] v = new float[128];
      for (int j = 0; j < 128; j++) {
        v[j] = rng.nextFloat();
      }
      predecessorMetaStore.put(ord, new Document(id, v, null, java.util.Map.of()));
    }

    // Build the staged batch: batchSize fresh ids + Documents.
    stagedIds = new String[batchSize];
    stagedDocs = new Document[batchSize];
    for (int i = 0; i < batchSize; i++) {
      String id = "staged-" + i;
      stagedIds[i] = id;
      float[] v = new float[128];
      for (int j = 0; j < 128; j++) {
        v[j] = rng.nextFloat();
      }
      stagedDocs[i] = new Document(id, v, null, java.util.Map.of());
    }

    // Build the read workload: READ_COUNT random ordinals + corresponding ids spread across
    // the predecessor (mostly) plus a couple in the new batch (the realistic top-K mix).
    readOrdinals = new int[READ_COUNT];
    readIds = new String[READ_COUNT];
    for (int i = 0; i < READ_COUNT; i++) {
      int ord = rng.nextInt(liveSize);
      readOrdinals[i] = ord;
      readIds[i] = "doc-" + ord;
    }
  }

  /**
   * Measures pure {@code copyOf} cost for both structures at the predecessor's size. This is the
   * O(N) hashmap-bulk-copy that we're trying to eliminate.
   */
  @Benchmark
  public void copyOfOnly(Blackhole bh) {
    bh.consume(InMemoryIdMapper.copyOf(predecessorIdMapper));
    bh.consume(InMemoryMetadataStore.copyOf(predecessorMetaStore));
  }

  /** Measures the realistic per-commit work: copyOf + bulk add of the staged batch. */
  @Benchmark
  public void copyOfAndStage(Blackhole bh) {
    InMemoryIdMapper m = InMemoryIdMapper.copyOf(predecessorIdMapper);
    InMemoryMetadataStore s = InMemoryMetadataStore.copyOf(predecessorMetaStore);
    for (int i = 0; i < batchSize; i++) {
      int ord = m.put(stagedIds[i]);
      s.put(ord, stagedDocs[i]);
    }
    bh.consume(m);
    bh.consume(s);
  }

  /**
   * Measures the realistic per-commit work plus a top-K read against the successor. This is the
   * end-to-end per-commit pattern as seen by a search() call landing on the new generation.
   */
  @Benchmark
  public void copyOfStageAndRead(Blackhole bh) {
    InMemoryIdMapper m = InMemoryIdMapper.copyOf(predecessorIdMapper);
    InMemoryMetadataStore s = InMemoryMetadataStore.copyOf(predecessorMetaStore);
    for (int i = 0; i < batchSize; i++) {
      int ord = m.put(stagedIds[i]);
      s.put(ord, stagedDocs[i]);
    }
    for (int i = 0; i < READ_COUNT; i++) {
      bh.consume(s.get(readOrdinals[i]));
      bh.consume(m.idOf(readOrdinals[i]));
    }
    bh.consume(m);
    bh.consume(s);
  }

  /** Forward-lookup read cost on a fully-built mapper at {@code liveSize}. */
  @Benchmark
  public void forwardLookup(Blackhole bh) {
    for (int i = 0; i < READ_COUNT; i++) {
      bh.consume(predecessorIdMapper.ordinalOf(readIds[i]));
    }
  }

  /** Reverse-lookup read cost (ordinal -> id) on a fully-built mapper. */
  @Benchmark
  public void reverseLookup(Blackhole bh) {
    for (int i = 0; i < READ_COUNT; i++) {
      bh.consume(predecessorIdMapper.idOf(readOrdinals[i]));
    }
  }

  /** Metadata read cost on a fully-built store. */
  @Benchmark
  public void metadataLookup(Blackhole bh) {
    for (int i = 0; i < READ_COUNT; i++) {
      bh.consume(predecessorMetaStore.get(readOrdinals[i]));
    }
  }
}
