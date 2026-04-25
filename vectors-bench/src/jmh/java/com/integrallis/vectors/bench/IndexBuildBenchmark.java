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
import com.integrallis.vectors.db.index.HnswIndexAdapter;
import com.integrallis.vectors.db.index.VamanaIndexAdapter;
import com.integrallis.vectors.ivf.IvfBuildParams;
import com.integrallis.vectors.ivf.IvfIndex;
import com.integrallis.vectors.quantization.ArrayVectorDataset;
import com.integrallis.vectors.quantization.ScalarBits;
import com.integrallis.vectors.quantization.ScalarQuantizer;
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

/**
 * JMH Tier-2: index construction baselines.
 *
 * <p>Measures the end-to-end build time for each index backend at representative corpus sizes. Uses
 * {@link Mode#SingleShotTime} because index builds are O(n·log n) — running them in a tight
 * throughput loop would be misleading and extremely slow at larger {@code n}.
 *
 * <p>Each benchmark invocation regenerates the corpus in {@link #setUp()} so that the timed call
 * measures only the build itself, not the allocation.
 *
 * <p>Run all build benchmarks:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=IndexBuildBenchmark
 * }</pre>
 */
@BenchmarkMode(Mode.SingleShotTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 2)
@Measurement(iterations = 5)
public class IndexBuildBenchmark {

  @Param({"1000", "5000"})
  public int n;

  @Param({"128"})
  public int dim;

  private float[][] corpus;
  private ArrayVectorDataset dataset;

  // HNSW params
  private static final int HNSW_M = 16;
  private static final int HNSW_EF = 200;

  // Vamana params
  private static final int VAMANA_R = 64;
  private static final int VAMANA_L = 128;
  private static final float VAMANA_ALPHA = 1.2f;

  // IVF params — K ≈ sqrt(n); nprobe is a search-time parameter, not needed here
  private int ivfK;

  @Setup(Level.Invocation)
  public void setUp() {
    Random rng = new Random(99L);
    corpus = new float[n][dim];
    for (float[] row : corpus) {
      for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    }
    dataset = new ArrayVectorDataset(corpus);
    ivfK = (int) Math.max(4, Math.sqrt(n));
  }

  // --- HNSW ---

  /**
   * Builds an in-memory HNSW graph with M={@value #HNSW_M} and efConstruction={@value #HNSW_EF}.
   */
  @Benchmark
  public HnswIndexAdapter buildHnsw() {
    HnswIndexAdapter adapter = new HnswIndexAdapter(HNSW_M, HNSW_EF);
    adapter.build(corpus, SimilarityFunction.EUCLIDEAN);
    return adapter;
  }

  // --- Vamana ---

  /**
   * Builds a Vamana DiskANN-style graph with R={@value #VAMANA_R}, L={@value #VAMANA_L},
   * alpha={@value #VAMANA_ALPHA}.
   */
  @Benchmark
  public VamanaIndexAdapter buildVamana() {
    VamanaIndexAdapter adapter = new VamanaIndexAdapter(VAMANA_R, VAMANA_L, VAMANA_ALPHA, 42L);
    adapter.build(corpus, SimilarityFunction.EUCLIDEAN);
    return adapter;
  }

  // --- IVF_FLAT ---

  /**
   * Builds an IVF-flat index: K-Means (K ≈ √n) followed by cluster assignment. Includes the
   * overhead of BuoyIndex training (the routing layer).
   */
  @Benchmark
  public IvfIndex buildIvfFlat() {
    IvfBuildParams params = new IvfBuildParams(ivfK, 30, 0f, false, 42L, 0);
    return IvfIndex.build(corpus, null, SimilarityFunction.EUCLIDEAN, params);
  }

  // --- Scalar Quantizer (SQ8) ---

  /**
   * Trains an SQ8 scalar quantizer and encodes the full corpus. The encode step is included because
   * in production both must happen before the index is queryable.
   */
  @Benchmark
  public Object buildSq8AndEncodeAll() {
    ScalarQuantizer sq = ScalarQuantizer.train(dataset, ScalarBits.INT8);
    return sq.encodeAll(dataset);
  }
}
