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

import com.integrallis.vectors.cache.SemanticCache;
import com.integrallis.vectors.cache.semantic.PayloadCodec;
import com.integrallis.vectors.cache.semantic.VectorDbSemanticCache;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import java.util.ArrayList;
import java.util.List;
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
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * P1.6 — JMH benchmark for {@link VectorDbSemanticCache} write path: the commit-storm contrast that
 * motivated the P0.6 {@code putAll} batch API. Each {@code putAll(batch)} performs exactly one
 * {@code collection.commit()} (one generation install), so {@code @Param batchSize} sweeps the
 * commit amortisation: {@code batchSize=1} is one commit per entry (the storm), {@code 100}/{@code
 * 1000} collapse the same insert volume to one commit per N entries.
 *
 * <p>{@link Mode#AverageTime} reports time per {@code putAll} (i.e. per commit); the {@code
 * cache-baseline.txt} doc derives commits/hour and the collapse factor. To isolate commit cost from
 * corpus growth, the benchmark upserts an existing key set (constant corpus size) — {@code putAll}
 * upserts by id, so re-putting the same keys re-commits without growing the collection. In-memory
 * FLAT/COSINE collection; no model, no network, no disk/S3 backend.
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=SemanticCachePutBenchmark
 * }</pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class SemanticCachePutBenchmark {

  private static final int DIM = 128;
  private static final int CORPUS = 2000;

  @Param({"1", "100", "1000"})
  public int batchSize;

  private VectorCollection collection;
  private VectorDbSemanticCache<String> cache;
  private float[][] embeddings;
  private int cursor;

  @Setup(Level.Trial)
  public void setUp() {
    Random rng = new Random(42L);
    collection =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .build();
    cache =
        VectorDbSemanticCache.builder(collection, PayloadCodec.identity()).threshold(0.92).build();

    embeddings = new float[CORPUS][];
    List<SemanticCache.Entry<String>> seed = new ArrayList<>(CORPUS);
    for (int i = 0; i < CORPUS; i++) {
      embeddings[i] = unitVec(rng);
      seed.add(new SemanticCache.Entry<>("k" + i, embeddings[i], "payload-" + i));
    }
    cache.putAll(seed); // initial population + one commit
  }

  @TearDown(Level.Trial)
  public void tearDown() {
    cache.close();
    collection.close();
  }

  private static float[] unitVec(Random rng) {
    float[] v = new float[DIM];
    double norm = 0;
    for (int i = 0; i < DIM; i++) {
      v[i] = rng.nextFloat() * 2f - 1f;
      norm += (double) v[i] * v[i];
    }
    float inv = (float) (1.0 / Math.sqrt(norm));
    for (int i = 0; i < DIM; i++) {
      v[i] *= inv;
    }
    return v;
  }

  /** One {@code putAll} of {@code batchSize} upserts of existing keys == exactly one commit. */
  @Benchmark
  public void putBatchCommit() {
    List<SemanticCache.Entry<String>> batch = new ArrayList<>(batchSize);
    for (int j = 0; j < batchSize; j++) {
      int idx = cursor;
      cursor = (cursor + 1) % CORPUS;
      batch.add(new SemanticCache.Entry<>("k" + idx, embeddings[idx], "payload-" + idx));
    }
    cache.putAll(batch);
  }
}
