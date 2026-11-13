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

import com.integrallis.vectors.core.cluster.EMeans;
import com.integrallis.vectors.core.cluster.KMeans;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Head-to-head benchmark: e-means (5 epochs) vs Lloyd's k-means (50 iterations) on identical
 * BuoyIndex-style workloads.
 *
 * <p>Hypothesis: e-means achieves same or lower quantization error in fewer epochs and less wall
 * time, because its mini-batch EMA update converges faster than Lloyd's full-batch mean
 * recomputation.
 *
 * <p>Run:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=KMeansEMeansBenchmark
 * }</pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 3, time = 1)
public class KMeansEMeansBenchmark {

  @Param({"10000", "50000", "100000"})
  int n;

  @Param({"64", "128"})
  int dim;

  @Param({"64", "256", "1024"})
  int k;

  private float[][] dataset;

  @Setup
  public void setUp() {
    Random rng = new Random(77L);
    dataset = new float[n][dim];
    for (float[] row : dataset) {
      for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    }
  }

  @Benchmark
  public float[][] trainKMeans() {
    return KMeans.train(dataset, k, 50, 42L);
  }

  @Benchmark
  public float[][] trainEMeans() {
    return EMeans.train(dataset, k, 5.0f, 42L);
  }
}
