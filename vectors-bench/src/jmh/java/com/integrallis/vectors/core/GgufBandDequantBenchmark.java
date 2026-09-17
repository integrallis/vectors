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
package com.integrallis.vectors.core;

import java.lang.foreign.MemorySegment;
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
 * Component ablation for the band GEMM arm: dequantization alone, single thread, one op = every row
 * of the matrix dequantized in {@code GgufBandGemm.KC} column blocks into a cache-resident buffer
 * (exactly the band arm's per-call dequant work, minus the sweeps). Band total minus this is the
 * sweep cost. Throughput in F32 elements/s = rows * cols / (s/op).
 *
 * <p>{@code dequant} selects the K-quant dequantisation arm explicitly ({@link
 * GgufKQuantDequant.Arm}; ignored for Q8_0), so old and new dequant are compared in one JVM. The
 * setup line prints the effective arm, which differs from the requested one only for the fused arms
 * at 4 float lanes.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
public class GgufBandDequantBenchmark {

  @Param({"Q4_K", "Q6_K", "Q8_0"})
  String format;

  @Param({"8192x2560", "2560x8192"})
  String shape;

  @Param({"scalar", "simd-byte", "simd-int", "simd-split"})
  String dequant;

  private GgufKQuantDequant.Arm arm;

  private int formatCode;
  private int rows;
  private int cols;
  private MemorySegment weights;
  private float[] panel;
  private byte[] raw;

  @Setup(Level.Trial)
  public void setUp() {
    String[] dims = shape.split("x");
    rows = Integer.parseInt(dims[0]);
    cols = Integer.parseInt(dims[1]);
    formatCode =
        switch (format) {
          case "Q4_K" -> GgufBandGemm.Q4_K;
          case "Q6_K" -> GgufBandGemm.Q6_K;
          default -> GgufBandGemm.Q8_0;
        };
    int blockBytes = GgufBandGemm.blockBytes(formatCode);
    int blockSize = GgufBandGemm.blockSize(formatCode);
    byte[] blocks = new byte[rows * (cols / blockSize) * blockBytes];
    new Random(7L).nextBytes(blocks);
    weights = MemorySegment.ofArray(blocks);
    panel = new float[GgufBandGemm.KC];
    raw = new byte[GgufBandGemm.KC / blockSize * blockBytes];
    arm = GgufKQuantDequant.parse(dequant);
    System.out.printf(
        "band-dequant format=%s shape=%s dequant=%s effective=%s lanes=%d config=%s%n",
        format,
        shape,
        dequant,
        formatCode == GgufBandGemm.Q8_0 ? "n/a(Q8_0)" : GgufKQuantDequant.effective(arm).label,
        GgufKQuantDequant.LANES,
        GgufBatchedMatmulKernel.bandDequantConfiguration());
  }

  @Benchmark
  public float dequantizeMatrix() {
    int blockSize = GgufBandGemm.blockSize(formatCode);
    int blockBytes = GgufBandGemm.blockBytes(formatCode);
    long rowBytes = (long) (cols / blockSize) * blockBytes;
    float probe = 0.0f;
    for (int row = 0; row < rows; row++) {
      for (int kOff = 0; kOff < cols; kOff += GgufBandGemm.KC) {
        int kcb = Math.min(GgufBandGemm.KC, cols - kOff);
        GgufBandGemm.dequantize(
            arm,
            formatCode,
            weights,
            row * rowBytes + (long) (kOff / blockSize) * blockBytes,
            kcb / blockSize,
            raw,
            panel,
            0);
        probe += panel[kcb - 1];
      }
    }
    return probe;
  }
}
