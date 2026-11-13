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
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
 * JMH benchmark for single-document embed kernel shapes (short-input latency baseline).
 *
 * <p>Issue #185 deliverable: measures the Vectors kernel portion of {@code
 * GgufEmbeddingBackend.embed(String)} and {@code embedAll(List)} for typical embedding model shapes
 * and short token counts.
 *
 * <p>The shapes represent encoder transformer matmuls:
 *
 * <ul>
 *   <li><b>FFN up:</b> hidden_dim → intermediate_dim (e.g., 384→1536, 768→3072)
 *   <li><b>FFN down:</b> intermediate_dim → hidden_dim (e.g., 1536→384, 3072→768)
 *   <li><b>QKV:</b> hidden_dim → 3×hidden_dim (e.g., 384→1152, 768→2304)
 * </ul>
 *
 * <p>Token counts 1, 4, 8, 16, 32 represent short query lengths.
 *
 * <p><b>Batching is conditional on the scenario:</b>
 *
 * <ul>
 *   <li>{@link Scenario#EMBED_SINGLE} (1 document): matches {@code embed(String)} and the fallback
 *       path for MEAN/CLS pooling in {@code embedAll}.
 *   <li>{@link Scenario#EMBED_BATCH_LAST_TOKEN} (4 documents): matches the {@code embedAll} fast
 *       path when pooling is LAST_TOKEN, the backend supports batched hidden states, and the
 *       document count is within {@code maxBatchSize}. Models does NOT batch for MEAN or CLS
 *       pooling.
 * </ul>
 *
 * <p>Three weight formats are covered: Q4_K, Q6_K, Q8_0. Each is measured on both the INTEGER
 * (activation-quantized) and BAND_F32 (dequantize-to-F32) routes where applicable. The existing
 * {@link GgufBatchedMatmulKernel#routingReport()} proves which arm executed after each fork.
 *
 * <p><b>Out of scope:</b> tokenization, pooling, normalization, KV-cache management — all
 * Models-owned.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class EmbedKernelShapesBenchmark {

  /** Weight format being benchmarked. */
  public enum Format {
    Q4_K(256, 210, 24),
    Q6_K(256, 210, 24),
    Q8_0(32, 34, 2);

    final int blockSize;
    final int blockBytes;
    final int scaleOffset;

    Format(int blockSize, int blockBytes, int scaleOffset) {
      this.blockSize = blockSize;
      this.blockBytes = blockBytes;
      this.scaleOffset = scaleOffset;
    }
  }

  /** Matmul role in an encoder forward pass. */
  public enum Role {
    FFN_UP(384, 1536, "MiniLM FFN up-projection 384→1536"),
    FFN_DOWN(1536, 384, "MiniLM FFN down-projection 1536→384"),
    QKV(384, 1152, "MiniLM QKV projection 384→1152"),
    FFN_UP_LARGE(768, 3072, "Standard FFN up-projection 768→3072"),
    FFN_DOWN_LARGE(3072, 768, "Standard FFN down-projection 3072→768"),
    QKV_LARGE(768, 2304, "Standard QKV projection 768→2304");

    final int cols;
    final int rows;
    final String description;

    Role(int cols, int rows, String description) {
      this.cols = cols;
      this.rows = rows;
      this.description = description;
    }
  }

  /**
   * Embedding scenario. In Models, batching is conditional:
   *
   * <ul>
   *   <li>{@code EMBED_SINGLE}: one document, {@code tokenCount} tokens. Matches {@code
   *       GgufEmbeddingBackend.embed(String)} and the fallback path for MEAN/CLS pooling in {@code
   *       embedAll}.
   *   <li>{@code EMBED_BATCH_LAST_TOKEN}: four documents, {@code tokenCount} tokens each. Matches
   *       the {@code GgufEmbeddingBackend.embedAll} fast path when pooling is LAST_TOKEN, the
   *       backend supports batched hidden states, and the document count is within {@code
   *       maxBatchSize}.
   * </ul>
   */
  public enum Scenario {
    EMBED_SINGLE(1, "single-document embed; also MEAN/CLS fallback"),
    EMBED_BATCH_LAST_TOKEN(4, "embedAll LAST_TOKEN batched");

    final int docCount;
    final String description;

    Scenario(int docCount, String description) {
      this.docCount = docCount;
      this.description = description;
    }
  }

  @Param({"Q4_K", "Q6_K", "Q8_0"})
  Format format;

  @Param({"FFN_UP", "FFN_DOWN", "QKV", "FFN_UP_LARGE", "FFN_DOWN_LARGE", "QKV_LARGE"})
  Role role;

  @Param({"1", "4", "8", "16", "32"})
  int tokenCount;

  @Param({"EMBED_SINGLE", "EMBED_BATCH_LAST_TOKEN"})
  Scenario scenario;

  @Param({"INTEGER", "BAND_F32"})
  String kernelMode;

  private float[] queries;
  private MemorySegment weights;
  private float[] out;

  // Quantization scratch (INTEGER route only)
  private byte[] q8Quants;
  private float[] q8Scales;
  private short[] q8Sums;

  @Setup(Level.Trial)
  public void setUp() {
    int cols = role.cols;
    int rows = role.rows;
    int docs = scenario.docCount;
    int totalBatch = tokenCount * docs;

    Random random = new Random(42L + format.ordinal() * 100L + role.ordinal());
    queries = new float[totalBatch * cols];
    for (int i = 0; i < queries.length; i++) {
      queries[i] = random.nextFloat() * 2.0f - 1.0f;
    }

    int blocksPerRow = cols / format.blockSize;
    int totalBlocks = rows * blocksPerRow;
    byte[] blocks = new byte[totalBlocks * format.blockBytes];
    random.nextBytes(blocks);
    ByteBuffer buffer = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
    short scale = Float.floatToFloat16(0.01f);
    for (int b = 0; b < totalBlocks; b++) {
      int offset = b * format.blockBytes + format.scaleOffset;
      if (offset + Short.BYTES <= blocks.length) {
        buffer.putShort(offset, scale);
      }
    }
    weights = MemorySegment.ofArray(blocks);
    out = new float[totalBatch * rows];

    q8Quants = new byte[totalBatch * cols];
    q8Scales = new float[totalBatch * (cols / format.blockSize)];
    if (format == Format.Q4_K) {
      q8Sums = new short[totalBatch * (cols / format.blockSize)];
    }

    // Reset routing counts so the per-fork report is clean
    GgufBatchedMatmulKernel.resetRoutingCounts();
  }

  @Benchmark
  public void embedKernel(Blackhole blackhole) {
    GgufBatchedMatmulKernel kernel =
        kernelMode.equals("BAND_F32")
            ? GgufBatchedMatmulKernel.BAND_F32
            : GgufBatchedMatmulKernel.INTEGER;

    int docs = scenario.docCount;
    int totalBatch = tokenCount * docs;

    switch (format) {
      case Q4_K -> {
        if (q8Sums == null) {
          q8Sums = new short[totalBatch * (role.cols / format.blockSize)];
        }
        VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
            queries,
            weights,
            totalBatch,
            role.rows,
            role.cols,
            out,
            q8Quants,
            q8Scales,
            q8Sums,
            kernel);
      }
      case Q6_K ->
          VectorUtil.ggufQ6_KQ8_KBatchedMatmul(
              queries, weights, totalBatch, role.rows, role.cols, out, q8Quants, q8Scales, kernel);
      case Q8_0 ->
          VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
              queries, weights, totalBatch, role.rows, role.cols, out, q8Quants, q8Scales, kernel);
    }
    blackhole.consume(out);
  }

  @TearDown(Level.Iteration)
  public void tearDown() {
    // Force routing report print after each iteration so we can verify arm selection
    // This is a no-op cost-wise compared to the matmul; it just prints to stderr
    if (System.getProperty("vectors.gguf.batchedMatmulKernel.reportPerIteration") != null) {
      System.err.println(GgufBatchedMatmulKernel.routingReport());
    }
  }
}
