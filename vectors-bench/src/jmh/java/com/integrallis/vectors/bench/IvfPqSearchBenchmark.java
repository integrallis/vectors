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
import com.integrallis.vectors.ivf.IvfBuildParams;
import com.integrallis.vectors.ivf.IvfIndex;
import com.integrallis.vectors.ivf.IvfSearchRequest;
import com.integrallis.vectors.ivf.IvfSearchResult;
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
 * JMH benchmark for IVF-PQ search latency across the primary recall / throughput knobs.
 *
 * <p>Parameter sweep:
 *
 * <ul>
 *   <li>{@code n} -- corpus size
 *   <li>{@code dim} -- vector dimension (M is chosen so {@code dim / M} is integral)
 *   <li>{@code pqSubspaces} -- number of PQ subspaces (must divide {@code dim})
 *   <li>{@code rescoreFactor} -- wide-heap multiplier for full-precision rescoring
 * </ul>
 *
 * <p>With {@code rescoreFactor = 1} the search path is pure ADC (byte-code table lookups); with
 * {@code rescoreFactor > 1} the wide heap is re-ranked against full-precision vectors to recover
 * recall at the cost of extra scoring work.
 *
 * <p>Run:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=IvfPqSearchBenchmark
 * }</pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class IvfPqSearchBenchmark {

  /** Dataset size (corpus vectors). */
  @Param({"10000", "100000"})
  int n;

  /**
   * Embedding dimension. Kept to {@code 128} and {@code 768} so the sweep over {@code pqSubspaces}
   * always yields a clean {@code dim / M} split (8, 16, 32 all divide both).
   */
  @Param({"128", "768"})
  int dim;

  /** PQ subspaces (M). Drives compression ratio and ADC table cost. */
  @Param({"8", "16", "32"})
  int pqSubspaces;

  /**
   * Rescore factor. {@code 1} = pure ADC scoring (fastest, lowest recall); {@code 4} = 4× wide heap
   * re-ranked against full-precision vectors (slower, higher recall).
   */
  @Param({"1", "4"})
  int rescoreFactor;

  private IvfIndex index;
  private float[] query;
  private int nprobe;

  @Setup(Level.Trial)
  public void setUp() {
    Random rng = new Random(77L);
    float[][] corpus = new float[n][dim];
    for (float[] row : corpus) {
      for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    }
    query = corpus[rng.nextInt(n)].clone();

    int k = (int) Math.max(4, Math.ceil(Math.sqrt(n)));
    // Probe √k clusters — the textbook IVF operating point at which recall plateaus.
    this.nprobe = (int) Math.max(1, Math.ceil(Math.sqrt(k)));
    IvfBuildParams base = new IvfBuildParams(k, 30, 0f, false, 42L, 0);
    IvfBuildParams params = base.withPq(pqSubspaces);
    index = IvfIndex.build(corpus, null, SimilarityFunction.EUCLIDEAN, params);
  }

  /**
   * Searches the IVF-PQ index. Each invocation rebuilds the per-probe ADC tables, scores {@code
   * nprobe} partitions, and (when {@code rescoreFactor > 1}) re-ranks the wide candidate heap
   * against full-precision vectors.
   */
  @Benchmark
  public IvfSearchResult searchIvfPq(Blackhole bh) {
    IvfSearchRequest req = IvfSearchRequest.of(query, 10, nprobe, rescoreFactor);
    IvfSearchResult result = index.search(req);
    bh.consume(result);
    return result;
  }
}
