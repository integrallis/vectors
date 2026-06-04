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
import java.util.Optional;
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
 * P1.6 — JMH benchmark for {@link VectorDbSemanticCache} read path: similarity-threshold {@code
 * lookup()} latency and RPS over an in-memory FLAT/COSINE {@link VectorCollection}.
 *
 * <p>Read-only (the corpus is built once in trial setup), so commit cost never contaminates these
 * numbers — the write/commit-storm contrast lives in {@link SemanticCachePutBenchmark}. Two queries
 * are measured: a deterministically-constructed near-duplicate of a stored vector (clears the 0.92
 * cosine threshold → hit, exercises payload decode) and a fresh random unit vector far from the
 * sparse high-dimensional corpus (→ miss, exercises top-1 search + threshold reject). No embedding
 * model or network is involved. Both percentiles ({@link Mode#SampleTime}, {@code ·p0.50}/{@code
 * ·p0.99} sub-rows) and RPS ({@link Mode#Throughput}) are reported.
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=SemanticCacheBenchmark
 * }</pre>
 */
@BenchmarkMode({Mode.Throughput, Mode.SampleTime})
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class SemanticCacheBenchmark {

  private static final int DIM = 128;
  private static final double THRESHOLD = 0.92;

  @Param({"1000", "5000"})
  public int corpus;

  private VectorCollection collection;
  private VectorDbSemanticCache<String> cache;
  private float[] hitQuery;
  private float[] missQuery;

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
        VectorDbSemanticCache.builder(collection, PayloadCodec.identity())
            .threshold(THRESHOLD)
            .build();

    List<SemanticCache.Entry<String>> entries = new ArrayList<>(corpus);
    float[] first = null;
    for (int i = 0; i < corpus; i++) {
      float[] v = unitVec(rng);
      if (i == 0) {
        first = v.clone();
      }
      entries.add(new SemanticCache.Entry<>("k" + i, v, "payload-" + i));
    }
    cache.putAll(entries);

    // Hit query: a 95%/5% blend of the first stored vector and noise -> cosine ~0.999 >= 0.92.
    hitQuery = blendUnit(first, unitVec(rng), 0.95f);
    // Miss query: a fresh random unit vector; in 128-d the nearest of a few thousand random unit
    // vectors has cosine well below 0.92, so this reliably falls under the threshold.
    missQuery = unitVec(rng);
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

  private static float[] blendUnit(float[] base, float[] noise, float w) {
    float[] v = new float[DIM];
    double norm = 0;
    for (int i = 0; i < DIM; i++) {
      v[i] = w * base[i] + (1f - w) * noise[i];
      norm += (double) v[i] * v[i];
    }
    float inv = (float) (1.0 / Math.sqrt(norm));
    for (int i = 0; i < DIM; i++) {
      v[i] *= inv;
    }
    return v;
  }

  @Benchmark
  public Optional<SemanticCache.Hit<String>> lookupHit() {
    return cache.lookup(hitQuery);
  }

  @Benchmark
  public Optional<SemanticCache.Hit<String>> lookupMiss() {
    return cache.lookup(missQuery);
  }
}
