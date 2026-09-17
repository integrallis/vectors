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

import com.integrallis.vectors.core.GgufBatchedMatmulKernel;
import com.integrallis.vectors.core.VectorUtil;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
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
import org.openjdk.jmh.infra.Blackhole;

/**
 * A/B of the GGUF batched matmul arithmetic: the established integer kernels (Q8 activations,
 * int8/int16 lanes) against the experimental dequantise-to-F32 band GEMM.
 *
 * <p>Both arms go through the same public {@link VectorUtil} entry point with an explicit {@link
 * GgufBatchedMatmulKernel}, so validation, threading policy ({@code vectors.gguf.parallel}, {@code
 * vectors.gguf.threads}) and the output contract are shared. The setup prints the runtime
 * capabilities so each fork records the ISA width, executor and band tile it actually ran with.
 *
 * <p>{@code shape} is {@code rowsxcols} (weight matrix rows by columns). Defaults cover the shape
 * of the existing {@code GgufQ4K/Q6K/Q8BatchedMatmulBenchmark} classes and Granite 4.1 3B's FFN
 * projections (hidden 2560, FFN 8192): up/gate {@code 8192x2560} and down {@code 2560x8192}. {@code
 * storage=mapped} maps a temporary file like a GGUF checkpoint; {@code heap} matches the existing
 * benchmarks.
 *
 * <p>To read the result as memory bandwidth, multiply ops/s by the quantized matrix size printed in
 * the setup line ({@code weightBytes}) and compare with {@code MemoryBandwidthBenchmark}.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
public class GgufBandGemmAbBenchmark {

  @Param({"Q4_K", "Q6_K", "Q8_0"})
  String format;

  @Param({"INTEGER", "BAND_F32"})
  String kernel;

  @Param({"1024x2048", "8192x2560", "2560x8192"})
  String shape;

  @Param({"1", "32", "512"})
  int batchSize;

  @Param({"heap"})
  String storage;

  /**
   * Label of the band arm's K-quant dequant arm this fork must be running ({@code
   * -Dvectors.gguf.band.dequant}, a JVM-wide constant). {@code any} skips the check; any other
   * value fails the trial unless the fork reports that requested arm, so a result can never carry
   * the wrong label.
   */
  @Param({"any"})
  String dequant;

  private int rows;
  private int cols;
  private GgufBatchedMatmulKernel arithmetic;
  private float[] queries;
  private float[] out;
  private byte[] q8Quants;
  private float[] q8Scales;
  private short[] q8Sums;
  private MemorySegment weights;
  private Arena arena;
  private Path mappedFile;

  @Setup(Level.Trial)
  public void setUp() throws IOException {
    String[] dims = shape.split("x");
    rows = Integer.parseInt(dims[0]);
    cols = Integer.parseInt(dims[1]);
    arithmetic = GgufBatchedMatmulKernel.valueOf(kernel);
    String dequantConfiguration = GgufBatchedMatmulKernel.bandDequantConfiguration();
    if (!dequant.equals("any") && !dequantConfiguration.contains("requested=" + dequant + ",")) {
      throw new IllegalStateException(
          "dequant param " + dequant + " but this fork runs " + dequantConfiguration);
    }
    int blockSize = format.equals("Q8_0") ? 32 : 256;
    int blockBytes =
        switch (format) {
          case "Q4_K" -> 144;
          case "Q6_K" -> 210;
          case "Q8_0" -> 34;
          default -> throw new IllegalArgumentException(format);
        };
    Random random = new Random(42L);
    queries = new float[batchSize * cols];
    for (int i = 0; i < queries.length; i++) {
      queries[i] = random.nextFloat() * 2.0f - 1.0f;
    }
    out = new float[batchSize * rows];
    q8Quants = new byte[batchSize * cols];
    q8Scales = new float[batchSize * (cols / blockSize)];
    q8Sums = new short[batchSize * (cols / 16)];

    byte[] blocks = randomBlocks(random, rows * (cols / blockSize), blockBytes);
    switch (storage) {
      case "heap" -> weights = MemorySegment.ofArray(blocks);
      case "mapped" -> {
        mappedFile = Files.createTempFile("band-gemm-", ".bin");
        Files.write(mappedFile, blocks);
        arena = Arena.ofShared();
        try (FileChannel channel = FileChannel.open(mappedFile, StandardOpenOption.READ)) {
          weights = channel.map(FileChannel.MapMode.READ_ONLY, 0, blocks.length, arena);
        }
        weights.load();
      }
      default -> throw new IllegalArgumentException("storage " + storage);
    }
    System.out.printf(
        "band-gemm-ab format=%s kernel=%s rows=%d cols=%d batch=%d storage=%s weightBytes=%d"
            + " band=%s bandDequant=%s capabilities=%s%n",
        format,
        kernel,
        rows,
        cols,
        batchSize,
        storage,
        blocks.length,
        GgufBatchedMatmulKernel.bandConfiguration(),
        dequantConfiguration,
        VectorUtil.runtimeCapabilities());
  }

  private byte[] randomBlocks(Random random, int blockCount, int blockBytes) {
    byte[] blocks = new byte[blockCount * blockBytes];
    random.nextBytes(blocks);
    ByteBuffer buffer = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
    short scale = Float.floatToFloat16(0.01f);
    for (int offset = 0; offset < blocks.length; offset += blockBytes) {
      switch (format) {
        case "Q4_K" -> {
          buffer.putShort(offset, scale);
          buffer.putShort(offset + Short.BYTES, Float.floatToFloat16(0.005f));
        }
        case "Q6_K" -> buffer.putShort(offset + 208, scale);
        default -> buffer.putShort(offset, scale);
      }
    }
    return blocks;
  }

  @TearDown(Level.Trial)
  public void tearDown() throws IOException {
    if (arena != null) {
      arena.close();
    }
    if (mappedFile != null) {
      Files.deleteIfExists(mappedFile);
    }
  }

  @Benchmark
  public void matmul(Blackhole blackhole) {
    switch (format) {
      case "Q4_K" ->
          VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
              queries, weights, batchSize, rows, cols, out, q8Quants, q8Scales, q8Sums, arithmetic);
      case "Q6_K" ->
          VectorUtil.ggufQ6_KQ8_KBatchedMatmul(
              queries, weights, batchSize, rows, cols, out, q8Quants, q8Scales, arithmetic);
      default ->
          VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
              queries, weights, batchSize, rows, cols, out, q8Quants, q8Scales, arithmetic);
    }
    blackhole.consume(out);
  }
}
