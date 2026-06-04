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

import com.integrallis.vectors.cache.CaffeineVectorCache;
import com.integrallis.vectors.cache.NoOpVectorCache;
import com.integrallis.vectors.cache.VectorCache;
import com.integrallis.vectors.cache.jcache.JCacheVectorCache;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import javax.cache.Cache;
import javax.cache.CacheManager;
import javax.cache.Caching;
import javax.cache.configuration.MutableConfiguration;
import javax.cache.spi.CachingProvider;
import org.openjdk.jmh.annotations.AuxCounters;
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
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;

/**
 * P1.6 — JMH benchmark for the exact-match {@link VectorCache} implementations: {@link
 * CaffeineVectorCache}, {@link JCacheVectorCache}, and {@link NoOpVectorCache} as a zero-hit
 * baseline (selected by {@code @Param impl}).
 *
 * <p>Measures, in one run: steady-state RPS ({@link Mode#Throughput}) and tail-latency percentiles
 * ({@link Mode#SampleTime}; read the {@code ·p0.50}/{@code ·p0.99} sub-rows) for an always-hit
 * lookup and a realistic ~80%-hit mix, hit-rate ({@code @AuxCounters} on the mixed method,
 * cross-check against {@code stats().hitRate()}), and {@code getOrCompute} single-flight under
 * contention ({@code @Threads(16)}). All measured methods return the looked-up value so the JIT
 * cannot eliminate the work. Vectors are seeded-random {@code float[}{@value #DIM}{@code ]}; no
 * model or network is involved. Curated results live in {@code
 * vectors-bench/jmh-results/cache-baseline.txt}.
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=CacheBenchmark
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
public class CacheBenchmark {

  private static final int DIM = 128;
  private static final int CAPACITY = 10_000;
  private static final int PROBE_LEN = 1 << 16; // power of two for branch-free masking
  private static final int PROBE_MASK = PROBE_LEN - 1;
  private static final double MIXED_HIT_RATE = 0.8;
  private static final String HOT_KEY = "hot";

  @Param({"caffeine", "jcache", "noop"})
  public String impl;

  private VectorCache<String, float[]> cache;
  private CachingProvider jcacheProvider;
  private CacheManager jcacheManager;
  private String[] residentKeys; // CAPACITY keys guaranteed resident
  private String[] mixedProbe; // PROBE_LEN keys, MIXED_HIT_RATE fraction resident
  private float[] loaderValue;

  @Setup(Level.Trial)
  public void setUp() {
    Random rng = new Random(42L);
    loaderValue = randomVec(rng);
    cache = newCache(impl);

    residentKeys = new String[CAPACITY];
    for (int i = 0; i < CAPACITY; i++) {
      residentKeys[i] = "k" + i;
      cache.put(residentKeys[i], randomVec(rng));
    }
    cache.put(HOT_KEY, randomVec(rng));

    // Probe sequence: MIXED_HIT_RATE fraction resident (hit), remainder absent (miss), shuffled
    // once with a fixed seed so the hot loop is branch-free and deterministic across forks.
    mixedProbe = new String[PROBE_LEN];
    int hits = (int) Math.round(MIXED_HIT_RATE * PROBE_LEN);
    for (int i = 0; i < PROBE_LEN; i++) {
      mixedProbe[i] = i < hits ? residentKeys[i % CAPACITY] : "absent" + i;
    }
    shuffle(mixedProbe, new Random(99L));
  }

  @TearDown(Level.Trial)
  public void tearDown() throws Exception {
    cache.close();
    if (jcacheManager != null) {
      jcacheManager.close();
    }
    if (jcacheProvider != null) {
      jcacheProvider.close();
    }
  }

  private VectorCache<String, float[]> newCache(String which) {
    switch (which) {
      case "caffeine":
        return CaffeineVectorCache.<String, float[]>builder().maximumSize(CAPACITY + 16L).build();
      case "noop":
        return new NoOpVectorCache<>();
      case "jcache":
        jcacheProvider =
            Caching.getCachingProvider(
                "com.github.benmanes.caffeine.jcache.spi.CaffeineCachingProvider");
        jcacheManager = jcacheProvider.getCacheManager();
        jcacheManager.destroyCache("bench");
        Cache<String, float[]> backing =
            jcacheManager.createCache(
                "bench",
                new MutableConfiguration<String, float[]>()
                    .setTypes(String.class, float[].class)
                    .setStatisticsEnabled(false));
        return new JCacheVectorCache<>(backing);
      default:
        throw new IllegalArgumentException("unknown impl: " + which);
    }
  }

  private static float[] randomVec(Random rng) {
    float[] v = new float[DIM];
    for (int i = 0; i < DIM; i++) {
      v[i] = rng.nextFloat();
    }
    return v;
  }

  private static void shuffle(String[] a, Random rng) {
    for (int i = a.length - 1; i > 0; i--) {
      int j = rng.nextInt(i + 1);
      String t = a[i];
      a[i] = a[j];
      a[j] = t;
    }
  }

  /** Per-thread probe cursor; persists across iterations, masked so overflow is harmless. */
  @State(Scope.Thread)
  public static class Cursor {
    int i;
  }

  /**
   * Per-iteration hit/miss counters. JMH prints the raw {@code hits} and {@code total} columns;
   * hit-rate is derived as {@code hits/total} in the curated doc (a derived {@code hitRate()}
   * method here is mis-aggregated by AuxCounters across iterations, so the raw counters are the
   * source of truth — for the mixed run they land at the designed ~0.8).
   */
  @AuxCounters(AuxCounters.Type.EVENTS)
  @State(Scope.Thread)
  public static class HitRate {
    public long hits;
    public long total;
  }

  @Benchmark
  public Optional<float[]> lookupAlwaysHit(Cursor c) {
    String key = residentKeys[(c.i++ & Integer.MAX_VALUE) % CAPACITY];
    return cache.get(key);
  }

  @Benchmark
  public Optional<float[]> lookupMixed(Cursor c, HitRate hr) {
    String key = mixedProbe[c.i++ & PROBE_MASK];
    Optional<float[]> v = cache.get(key);
    hr.total++;
    if (v.isPresent()) {
      hr.hits++;
    }
    return v;
  }

  @Benchmark
  @Threads(16)
  public float[] getOrComputeContended() {
    return cache.getOrCompute(HOT_KEY, k -> loaderValue);
  }
}
