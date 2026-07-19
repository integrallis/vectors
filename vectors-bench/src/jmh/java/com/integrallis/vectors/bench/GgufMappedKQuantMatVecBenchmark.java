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

/** Measures the K-quant GEMV path used by file-mapped GGUF model weights. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
public class GgufMappedKQuantMatVecBenchmark {

  @Param("1024")
  int rows;

  @Param("2048")
  int cols;

  @Param("256")
  int auxiliaryRows;

  @Param("32")
  int batchSize;

  private Arena arena;
  private Path q4Path;
  private Path q5Path;
  private Path q6Path;
  private MemorySegment q4Weights;
  private MemorySegment q5Weights;
  private MemorySegment q6Weights;
  private float[] query;
  private float[] batchedQueries;
  private float[] out;
  private float[] batchedOut;
  private float[] secondOut;
  private float[] thirdOut;
  private byte[] q8Quants;
  private float[] q8Scales;
  private short[] q8Sums;
  private byte[] batchedQ8Quants;
  private float[] batchedQ8Scales;
  private short[] batchedQ8Sums;

  @Setup(Level.Trial)
  public void setUp() throws IOException {
    arena = Arena.ofShared();
    Random random = new Random(0x26_4bL);
    q4Path = Files.createTempFile("mapped-q4-k-", ".bin");
    q5Path = Files.createTempFile("mapped-q5-k-", ".bin");
    q6Path = Files.createTempFile("mapped-q6-k-", ".bin");
    q4Weights = mapReadOnly(randomBlocks(random, rows * (cols / 256) * 144, 144, 0, 2), q4Path);
    q5Weights = mapReadOnly(randomBlocks(random, rows * (cols / 256) * 176, 176, 0, 2), q5Path);
    q6Weights = mapReadOnly(randomBlocks(random, rows * (cols / 256) * 210, 210, 208), q6Path);
    if (!q4Weights.isMapped() || !q5Weights.isMapped() || !q6Weights.isMapped()) {
      throw new IllegalStateException("K-quant benchmark weights must be file mapped");
    }
    Thread worker = Thread.ofPlatform().unstarted(() -> {});
    if (!q4Weights.isAccessibleBy(worker)
        || !q5Weights.isAccessibleBy(worker)
        || !q6Weights.isAccessibleBy(worker)) {
      throw new IllegalStateException("K-quant benchmark weights must be worker accessible");
    }

    query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 2.0f - 1.0f;
    }
    batchedQueries = new float[batchSize * cols];
    for (int index = 0; index < batchedQueries.length; index++) {
      batchedQueries[index] = random.nextFloat() * 2.0f - 1.0f;
    }
    out = new float[rows];
    batchedOut = new float[batchSize * rows];
    secondOut = new float[rows];
    thirdOut = new float[Math.max(rows, auxiliaryRows)];
    q8Quants = new byte[cols];
    q8Scales = new float[cols / 256];
    q8Sums = new short[cols / 16];
    batchedQ8Quants = new byte[batchSize * cols];
    batchedQ8Scales = new float[batchSize * (cols / 256)];
    batchedQ8Sums = new short[batchSize * (cols / 16)];
  }

  @TearDown(Level.Trial)
  public void tearDown() throws IOException {
    arena.close();
    Files.deleteIfExists(q4Path);
    Files.deleteIfExists(q5Path);
    Files.deleteIfExists(q6Path);
  }

  @Benchmark
  public float[] q4K() {
    VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
        query, q4Weights, rows, cols, out, q8Quants, q8Scales, q8Sums);
    return out;
  }

  @Benchmark
  public float[] q5K() {
    VectorUtil.ggufQ5_KQ8_KBatchDotProduct(
        query, q5Weights, rows, cols, out, q8Quants, q8Scales, q8Sums);
    return out;
  }

  @Benchmark
  public float[] q6K() {
    VectorUtil.ggufQ6_KQ8_KBatchDotProduct(query, q6Weights, rows, cols, out, q8Quants, q8Scales);
    return out;
  }

  @Benchmark
  public float[] q4KBatched() {
    VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
        batchedQueries,
        q4Weights,
        batchSize,
        rows,
        cols,
        batchedOut,
        batchedQ8Quants,
        batchedQ8Scales,
        batchedQ8Sums);
    return batchedOut;
  }

  @Benchmark
  public void q4KDual(Blackhole blackhole) {
    VectorUtil.ggufQ4_KQ8_KDualBatchDotProduct(
        query, q4Weights, rows, out, q4Weights, rows, secondOut, cols, q8Quants, q8Scales, q8Sums);
    blackhole.consume(out);
    blackhole.consume(secondOut);
  }

  @Benchmark
  public void q4KQ4KQ6KTriple(Blackhole blackhole) {
    VectorUtil.ggufQ4_KQ4_KQ6_KQ8_KTripleBatchDotProduct(
        query,
        q4Weights,
        rows,
        out,
        q4Weights,
        auxiliaryRows,
        secondOut,
        q6Weights,
        auxiliaryRows,
        thirdOut,
        cols,
        q8Quants,
        q8Scales,
        q8Sums);
    blackhole.consume(out);
    blackhole.consume(secondOut);
    blackhole.consume(thirdOut);
  }

  private MemorySegment mapReadOnly(byte[] bytes, Path path) throws IOException {
    Files.write(path, bytes);
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      return channel.map(FileChannel.MapMode.READ_ONLY, 0, bytes.length, arena);
    }
  }

  private static byte[] randomBlocks(
      Random random, int byteCount, int blockBytes, int... scaleOffsets) {
    byte[] bytes = new byte[byteCount];
    random.nextBytes(bytes);
    ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    short scale = Float.floatToFloat16(0.01f);
    for (int block = 0; block < byteCount; block += blockBytes) {
      for (int scaleOffset : scaleOffsets) {
        buffer.putShort(block + scaleOffset, scale);
      }
    }
    return bytes;
  }
}
