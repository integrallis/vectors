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
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.Locale;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Experimental dequantise-to-F32 band GEMM for GGUF Q4_K, Q6_K and Q8_0 weights ({@link
 * GgufBatchedMatmulKernel#BAND_F32}).
 *
 * <p>The idea (not the code) comes from the Apache-2.0 qxotic JAM engine's {@code BandGemm}: a task
 * dequantizes a cache-sized <i>panel</i> of weight rows, restricted to one {@link #KC}-element
 * column block, into F32 scratch, then sweeps activation <i>tiles</i> through it with an {@code MR
 * x NR} register tile of vector FMA accumulators whose lanes run along the column axis. Each
 * accumulator is reduced once per column block. This implementation differs from qxotic's: rows and
 * activations stay in their natural row-major layout (no interleaved packing copy), scratch is heap
 * {@code float[]}, the reduction is a fixed-order lane sum (deterministic across interpreter and
 * JIT), and threading reuses the vectors-core GGUF executor with the integer kernels' parallelism
 * threshold so an A/B compares arithmetic, not scheduling policy.
 *
 * <p>Layout: {@code queries = [n][k]}, {@code out = [n][m]}, weights {@code [m][k]} quantized.
 *
 * <p>Knobs (all resolved once, all reported by {@link #describe()}): {@code
 * -Dvectors.gguf.band.tile=3x3|4x4}, {@code -Dvectors.gguf.band.kc=<multiple of 256>}, {@code
 * -Dvectors.gguf.band.panelKb=<KiB>}.
 */
final class GgufBandGemm {

  static final VectorSpecies<Float> SPECIES = PanamaVectorUtilSupport.FLOAT_SPECIES;
  static final int LANES = SPECIES.length();

  static final String TILE_PROPERTY = "vectors.gguf.band.tile";
  static final String KC_PROPERTY = "vectors.gguf.band.kc";
  static final String PANEL_KB_PROPERTY = "vectors.gguf.band.panelKb";

  /**
   * 4x4 holds 16 accumulators + 4 activation loads + 1 weight load = 21 live vectors: spill-free
   * only with 32 vector registers (AVX-512 ZMM on C2, AArch64 NEON). 3x3 holds 13 and fits AVX2's
   * 16 YMM. This default is register-count reasoning, not a measurement; the tile is a knob.
   */
  static final boolean WIDE_TILE = resolveWideTile(System.getProperty(TILE_PROPERTY));

  static final int MR = WIDE_TILE ? 4 : 3;
  static final int NR = MR;

  /** Column block in elements: a multiple of 256 so every K-quant and Q8_0 block stays whole. */
  static final int KC = resolveKc(System.getProperty(KC_PROPERTY));

  static final int PANEL_BYTES = resolvePositive(PANEL_KB_PROPERTY, 256) * 1024;

  /** Panel rows are whole bands and a multiple of 16 so neighbouring panels rarely share a line. */
  static final int PANEL_GRANULE = MR * 16 / gcd(MR, 16);

  static final int Q4_K = 0;
  static final int Q6_K = 1;
  static final int Q8_0 = 2;

  private static final ThreadLocal<Scratch> SCRATCH = ThreadLocal.withInitial(Scratch::new);

  private GgufBandGemm() {}

  static String describe() {
    return "band-f32(tile="
        + MR
        + "x"
        + NR
        + ",kc="
        + KC
        + ",panelKb="
        + PANEL_BYTES / 1024
        + ",lanes="
        + LANES
        + ")";
  }

  static void ggufQ4_K(float[] queries, MemorySegment weights, int n, int m, int k, float[] out) {
    gemm(Q4_K, queries, weights, n, m, k, out);
  }

  static void ggufQ6_K(float[] queries, MemorySegment weights, int n, int m, int k, float[] out) {
    gemm(Q6_K, queries, weights, n, m, k, out);
  }

  static void ggufQ8_0(float[] queries, MemorySegment weights, int n, int m, int k, float[] out) {
    gemm(Q8_0, queries, weights, n, m, k, out);
  }

  static int blockSize(int format) {
    return format == Q8_0 ? VectorUtilSupport.GGUF_Q_BLOCK_SIZE : 256;
  }

  static int blockBytes(int format) {
    return switch (format) {
      case Q4_K -> VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES;
      case Q6_K -> VectorUtilSupport.GGUF_Q6_K_BLOCK_BYTES;
      case Q8_0 -> VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES;
      default -> throw new IllegalArgumentException("format " + format);
    };
  }

  private static long formatMinElements(int format) {
    return format == Q8_0
        ? GgufParallelSupport.Q8_MIN_ELEMENTS
        : GgufParallelSupport.DEFAULT_MIN_ELEMENTS;
  }

  static void gemm(
      int format, float[] queries, MemorySegment weights, int n, int m, int k, float[] out) {
    Arrays.fill(out, 0, n * m, 0.0f);
    if (k == 0 || m == 0 || n == 0) {
      return;
    }
    Plan plan = new Plan(format, weights, n, m, k);
    GgufParallelSupport.forEachTask(
        plan.parallel, plan.tasks, task -> runTask(plan, queries, weights, out, task));
  }

  /** The blocking of one call: column blocks, panels, token splits and task numbering. */
  static final class Plan {
    final int format;
    final int n;
    final int m;
    final int k;
    final int kc;
    final int blockSize;
    final int blockBytes;
    final long rowBytes;
    final int panelRows;
    final int panels;
    final int fullTiles;
    final int splits;
    final int tilesPerSplit;
    final int tasks;
    final boolean parallel;

    Plan(int format, MemorySegment weights, int n, int m, int k) {
      this.format = format;
      this.n = n;
      this.m = m;
      this.k = k;
      this.blockSize = blockSize(format);
      this.blockBytes = blockBytes(format);
      this.rowBytes = (long) (k / blockSize) * blockBytes;
      this.kc = Math.min(KC, k);
      this.parallel =
          GgufParallelSupport.willParallelize(weights, (long) m * k, formatMinElements(format));
      int threads = parallel ? GgufParallelSupport.parallelism() : 1;
      int byCache = PANEL_BYTES / (kc * Float.BYTES);
      int byBalance = parallel ? m / (4 * threads) : Integer.MAX_VALUE;
      this.panelRows =
          Math.max(PANEL_GRANULE, Math.min(byCache, byBalance) / PANEL_GRANULE * PANEL_GRANULE);
      this.panels = ceilDiv(m, panelRows);
      this.fullTiles = n / NR;
      int wanted = 4 * threads;
      this.splits =
          parallel && panels < wanted
              ? Math.max(1, Math.min(fullTiles, ceilDiv(wanted, panels)))
              : 1;
      this.tilesPerSplit = splits == 1 ? fullTiles : ceilDiv(fullTiles, splits);
      this.tasks = Math.multiplyExact(panels, splits);
    }
  }

  private static void runTask(
      Plan plan, float[] queries, MemorySegment weights, float[] out, int task) {
    int panel = task / plan.splits;
    int split = task % plan.splits;
    int tLo = split * plan.tilesPerSplit;
    int tHi = Math.min(plan.fullTiles, tLo + plan.tilesPerSplit);
    boolean leftovers = split == plan.splits - 1 && plan.fullTiles * NR < plan.n;
    if (tLo >= tHi && !leftovers) {
      return;
    }
    int m = plan.m;
    int k = plan.k;
    int r0 = panel * plan.panelRows;
    int bands = ceilDiv(Math.min(plan.panelRows, m - r0), MR);
    int bandRows = bands * MR;
    Scratch scratch = SCRATCH.get();
    float[] rowsF32 = scratch.panel(bandRows * plan.kc);
    byte[] rawBlocks = scratch.bytes((plan.kc / plan.blockSize) * plan.blockBytes);
    float[] acc = scratch.acc;

    for (int kOff = 0; kOff < k; kOff += plan.kc) {
      int kcb = Math.min(plan.kc, k - kOff);
      int firstBlock = kOff / plan.blockSize;
      int blocks = kcb / plan.blockSize;
      for (int i = 0; i < bandRows; i++) {
        int row = r0 + i;
        if (row < m) {
          long offset = row * plan.rowBytes + (long) firstBlock * plan.blockBytes;
          dequantize(plan.format, weights, offset, blocks, rawBlocks, rowsF32, i * kcb);
        } else {
          Arrays.fill(rowsF32, i * kcb, (i + 1) * kcb, 0.0f);
        }
      }
      for (int tile = tLo; tile < tHi; tile++) {
        int s0 = tile * NR;
        int a0 = s0 * k + kOff;
        for (int band = 0; band < bands; band++) {
          if (WIDE_TILE) {
            sweep44(rowsF32, band * MR * kcb, kcb, queries, a0, k, acc);
          } else {
            sweep33(rowsF32, band * MR * kcb, kcb, queries, a0, k, acc);
          }
          store(acc, NR, out, s0, NR, r0 + band * MR, m);
        }
      }
      if (leftovers) {
        for (int s = plan.fullTiles * NR; s < plan.n; s++) {
          int a0 = s * k + kOff;
          for (int band = 0; band < bands; band++) {
            if (WIDE_TILE) {
              sweep41(rowsF32, band * MR * kcb, kcb, queries, a0, acc);
            } else {
              sweep31(rowsF32, band * MR * kcb, kcb, queries, a0, acc);
            }
            store(acc, 1, out, s, 1, r0 + band * MR, m);
          }
        }
      }
    }
  }

  /**
   * Adds MR x {@code cols} accumulators (lanes summed in lane order) into {@code out[(s0 + j) * m +
   * row]}, skipping zero-padded rows past {@code m}.
   */
  private static void store(
      float[] acc, int stride, float[] out, int s0, int cols, int rowBase, int m) {
    int validRows = Math.min(MR, m - rowBase);
    for (int i = 0; i < validRows; i++) {
      for (int j = 0; j < cols; j++) {
        int base = (i * stride + j) * LANES;
        float sum = 0.0f;
        for (int lane = 0; lane < LANES; lane++) {
          sum += acc[base + lane];
        }
        out[(s0 + j) * m + rowBase + i] += sum;
      }
    }
  }

  // ---- register tiles: branch-free, call-free, every vector stays in one frame ----

  static void sweep33(float[] w, int w0, int kc, float[] a, int a0, int aStride, float[] acc) {
    int w1 = w0 + kc;
    int w2 = w1 + kc;
    int a1 = a0 + aStride;
    int a2 = a1 + aStride;
    FloatVector c00 = FloatVector.zero(SPECIES);
    FloatVector c01 = c00;
    FloatVector c02 = c00;
    FloatVector c10 = c00;
    FloatVector c11 = c00;
    FloatVector c12 = c00;
    FloatVector c20 = c00;
    FloatVector c21 = c00;
    FloatVector c22 = c00;
    for (int i = 0; i < kc; i += LANES) {
      FloatVector x0 = FloatVector.fromArray(SPECIES, a, a0 + i);
      FloatVector x1 = FloatVector.fromArray(SPECIES, a, a1 + i);
      FloatVector x2 = FloatVector.fromArray(SPECIES, a, a2 + i);
      FloatVector v = FloatVector.fromArray(SPECIES, w, w0 + i);
      c00 = v.fma(x0, c00);
      c01 = v.fma(x1, c01);
      c02 = v.fma(x2, c02);
      v = FloatVector.fromArray(SPECIES, w, w1 + i);
      c10 = v.fma(x0, c10);
      c11 = v.fma(x1, c11);
      c12 = v.fma(x2, c12);
      v = FloatVector.fromArray(SPECIES, w, w2 + i);
      c20 = v.fma(x0, c20);
      c21 = v.fma(x1, c21);
      c22 = v.fma(x2, c22);
    }
    c00.intoArray(acc, 0);
    c01.intoArray(acc, LANES);
    c02.intoArray(acc, 2 * LANES);
    c10.intoArray(acc, 3 * LANES);
    c11.intoArray(acc, 4 * LANES);
    c12.intoArray(acc, 5 * LANES);
    c20.intoArray(acc, 6 * LANES);
    c21.intoArray(acc, 7 * LANES);
    c22.intoArray(acc, 8 * LANES);
  }

  static void sweep44(float[] w, int w0, int kc, float[] a, int a0, int aStride, float[] acc) {
    int w1 = w0 + kc;
    int w2 = w1 + kc;
    int w3 = w2 + kc;
    int a1 = a0 + aStride;
    int a2 = a1 + aStride;
    int a3 = a2 + aStride;
    FloatVector c00 = FloatVector.zero(SPECIES);
    FloatVector c01 = c00;
    FloatVector c02 = c00;
    FloatVector c03 = c00;
    FloatVector c10 = c00;
    FloatVector c11 = c00;
    FloatVector c12 = c00;
    FloatVector c13 = c00;
    FloatVector c20 = c00;
    FloatVector c21 = c00;
    FloatVector c22 = c00;
    FloatVector c23 = c00;
    FloatVector c30 = c00;
    FloatVector c31 = c00;
    FloatVector c32 = c00;
    FloatVector c33 = c00;
    for (int i = 0; i < kc; i += LANES) {
      FloatVector x0 = FloatVector.fromArray(SPECIES, a, a0 + i);
      FloatVector x1 = FloatVector.fromArray(SPECIES, a, a1 + i);
      FloatVector x2 = FloatVector.fromArray(SPECIES, a, a2 + i);
      FloatVector x3 = FloatVector.fromArray(SPECIES, a, a3 + i);
      FloatVector v = FloatVector.fromArray(SPECIES, w, w0 + i);
      c00 = v.fma(x0, c00);
      c01 = v.fma(x1, c01);
      c02 = v.fma(x2, c02);
      c03 = v.fma(x3, c03);
      v = FloatVector.fromArray(SPECIES, w, w1 + i);
      c10 = v.fma(x0, c10);
      c11 = v.fma(x1, c11);
      c12 = v.fma(x2, c12);
      c13 = v.fma(x3, c13);
      v = FloatVector.fromArray(SPECIES, w, w2 + i);
      c20 = v.fma(x0, c20);
      c21 = v.fma(x1, c21);
      c22 = v.fma(x2, c22);
      c23 = v.fma(x3, c23);
      v = FloatVector.fromArray(SPECIES, w, w3 + i);
      c30 = v.fma(x0, c30);
      c31 = v.fma(x1, c31);
      c32 = v.fma(x2, c32);
      c33 = v.fma(x3, c33);
    }
    c00.intoArray(acc, 0);
    c01.intoArray(acc, LANES);
    c02.intoArray(acc, 2 * LANES);
    c03.intoArray(acc, 3 * LANES);
    c10.intoArray(acc, 4 * LANES);
    c11.intoArray(acc, 5 * LANES);
    c12.intoArray(acc, 6 * LANES);
    c13.intoArray(acc, 7 * LANES);
    c20.intoArray(acc, 8 * LANES);
    c21.intoArray(acc, 9 * LANES);
    c22.intoArray(acc, 10 * LANES);
    c23.intoArray(acc, 11 * LANES);
    c30.intoArray(acc, 12 * LANES);
    c31.intoArray(acc, 13 * LANES);
    c32.intoArray(acc, 14 * LANES);
    c33.intoArray(acc, 15 * LANES);
  }

  /** One activation column against a 3-row band (decode, and batch tails). */
  static void sweep31(float[] w, int w0, int kc, float[] a, int a0, float[] acc) {
    int w1 = w0 + kc;
    int w2 = w1 + kc;
    FloatVector c0 = FloatVector.zero(SPECIES);
    FloatVector c1 = c0;
    FloatVector c2 = c0;
    for (int i = 0; i < kc; i += LANES) {
      FloatVector x = FloatVector.fromArray(SPECIES, a, a0 + i);
      c0 = FloatVector.fromArray(SPECIES, w, w0 + i).fma(x, c0);
      c1 = FloatVector.fromArray(SPECIES, w, w1 + i).fma(x, c1);
      c2 = FloatVector.fromArray(SPECIES, w, w2 + i).fma(x, c2);
    }
    c0.intoArray(acc, 0);
    c1.intoArray(acc, LANES);
    c2.intoArray(acc, 2 * LANES);
  }

  /** One activation column against a 4-row band. */
  static void sweep41(float[] w, int w0, int kc, float[] a, int a0, float[] acc) {
    int w1 = w0 + kc;
    int w2 = w1 + kc;
    int w3 = w2 + kc;
    FloatVector c0 = FloatVector.zero(SPECIES);
    FloatVector c1 = c0;
    FloatVector c2 = c0;
    FloatVector c3 = c0;
    for (int i = 0; i < kc; i += LANES) {
      FloatVector x = FloatVector.fromArray(SPECIES, a, a0 + i);
      c0 = FloatVector.fromArray(SPECIES, w, w0 + i).fma(x, c0);
      c1 = FloatVector.fromArray(SPECIES, w, w1 + i).fma(x, c1);
      c2 = FloatVector.fromArray(SPECIES, w, w2 + i).fma(x, c2);
      c3 = FloatVector.fromArray(SPECIES, w, w3 + i).fma(x, c3);
    }
    c0.intoArray(acc, 0);
    c1.intoArray(acc, LANES);
    c2.intoArray(acc, 2 * LANES);
    c3.intoArray(acc, 3 * LANES);
  }

  // ---- dequantization: one bulk copy of the row's block run, then scalar loops C2 may vectorize
  // --

  /**
   * Dequantizes {@code blocks} whole blocks starting at byte {@code offset} of {@code weights} into
   * {@code dst[dst0 ..]}. Public to the package so the benchmark can time dequantization alone.
   */
  static void dequantize(
      int format,
      MemorySegment weights,
      long offset,
      int blocks,
      byte[] raw,
      float[] dst,
      int dst0) {
    int blockBytes = blockBytes(format);
    MemorySegment.copy(weights, ValueLayout.JAVA_BYTE, offset, raw, 0, blocks * blockBytes);
    switch (format) {
      case Q4_K -> {
        for (int block = 0; block < blocks; block++) {
          dequantizeQ4_KBlock(raw, block * blockBytes, dst, dst0 + block * 256);
        }
      }
      case Q6_K -> {
        for (int block = 0; block < blocks; block++) {
          dequantizeQ6_KBlock(raw, block * blockBytes, dst, dst0 + block * 256);
        }
      }
      case Q8_0 -> {
        for (int block = 0; block < blocks; block++) {
          dequantizeQ8_0Block(raw, block * blockBytes, dst, dst0 + block * 32);
        }
      }
      default -> throw new IllegalArgumentException("format " + format);
    }
  }

  static void dequantizeQ4_KBlock(byte[] raw, int bo, float[] dst, int o) {
    float d = float16(raw, bo);
    float dMin = float16(raw, bo + 2);
    int scales = bo + VectorUtilSupport.GGUF_Q4_K_SCALES_OFFSET;
    int quants = bo + VectorUtilSupport.GGUF_Q4_K_QUANTS_OFFSET;
    for (int pair = 0; pair < 4; pair++) {
      int q = quants + pair * 32;
      int lowGroup = 2 * pair;
      float lowScale = d * qKScale(raw, scales, lowGroup);
      float lowMin = dMin * qKMin(raw, scales, lowGroup);
      int lowOut = o + lowGroup * 32;
      for (int i = 0; i < 32; i++) {
        dst[lowOut + i] = lowScale * (raw[q + i] & 0x0F) - lowMin;
      }
      int highGroup = lowGroup + 1;
      float highScale = d * qKScale(raw, scales, highGroup);
      float highMin = dMin * qKMin(raw, scales, highGroup);
      int highOut = o + highGroup * 32;
      for (int i = 0; i < 32; i++) {
        dst[highOut + i] = highScale * ((raw[q + i] >> 4) & 0x0F) - highMin;
      }
    }
  }

  static void dequantizeQ6_KBlock(byte[] raw, int bo, float[] dst, int o) {
    float d =
        float16(
            raw,
            bo
                + VectorUtilSupport.GGUF_Q6_K_QL_BYTES
                + VectorUtilSupport.GGUF_Q6_K_QH_BYTES
                + VectorUtilSupport.GGUF_Q6_K_SCALES);
    int scales = bo + VectorUtilSupport.GGUF_Q6_K_QL_BYTES + VectorUtilSupport.GGUF_Q6_K_QH_BYTES;
    for (int half = 0; half < 2; half++) {
      int qhBase = bo + VectorUtilSupport.GGUF_Q6_K_QL_BYTES + half * 32;
      for (int quarter = 0; quarter < 4; quarter++) {
        int qlBase = bo + half * 64 + (quarter & 1) * 32;
        int lowShift = (quarter >> 1) * 4;
        int highShift = 2 * quarter;
        for (int sub = 0; sub < 2; sub++) {
          float scale = d * raw[scales + half * 8 + 2 * quarter + sub];
          int ql = qlBase + sub * 16;
          int qh = qhBase + sub * 16;
          int out = o + half * 128 + quarter * 32 + sub * 16;
          for (int i = 0; i < 16; i++) {
            int quant =
                ((raw[ql + i] >> lowShift) & 0x0F) | (((raw[qh + i] >> highShift) & 0x03) << 4);
            dst[out + i] = scale * (quant - 32);
          }
        }
      }
    }
  }

  static void dequantizeQ8_0Block(byte[] raw, int bo, float[] dst, int o) {
    float d = float16(raw, bo);
    int q = bo + Short.BYTES;
    for (int i = 0; i < VectorUtilSupport.GGUF_Q_BLOCK_SIZE; i++) {
      dst[o + i] = d * raw[q + i];
    }
  }

  private static float float16(byte[] raw, int offset) {
    return Float.float16ToFloat((short) ((raw[offset] & 0xFF) | (raw[offset + 1] << 8)));
  }

  private static int qKScale(byte[] raw, int scales, int group) {
    if (group < 4) {
      return raw[scales + group] & 0x3F;
    }
    return (raw[scales + group + 4] & 0x0F) | (((raw[scales + group - 4] & 0xFF) >>> 6) << 4);
  }

  private static int qKMin(byte[] raw, int scales, int group) {
    if (group < 4) {
      return raw[scales + group + 4] & 0x3F;
    }
    return ((raw[scales + group + 4] & 0xFF) >>> 4) | (((raw[scales + group] & 0xFF) >>> 6) << 4);
  }

  /** Per-thread, slot-stable scratch; grows on demand and is retained by the thread. */
  static final class Scratch {
    private float[] panel = new float[0];
    private byte[] bytes = new byte[0];
    final float[] acc = new float[16 * LANES];

    float[] panel(int size) {
      if (panel.length < size) {
        panel = new float[size];
      }
      return panel;
    }

    byte[] bytes(int size) {
      if (bytes.length < size) {
        bytes = new byte[size];
      }
      return bytes;
    }
  }

  static boolean resolveWideTile(String configured) {
    if (configured == null || configured.isBlank()) {
      String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
      return SPECIES.vectorBitSize() >= 512 || arch.equals("aarch64") || arch.equals("arm64");
    }
    return switch (configured.trim()) {
      case "3x3" -> false;
      case "4x4" -> true;
      default ->
          throw new IllegalArgumentException(
              "-D" + TILE_PROPERTY + " must be 3x3 or 4x4; got: " + configured);
    };
  }

  static int resolveKc(String configured) {
    if (configured == null || configured.isBlank()) {
      return 512;
    }
    int value = Integer.parseInt(configured.trim());
    if (value < 256 || value % 256 != 0) {
      throw new IllegalArgumentException(
          "-D" + KC_PROPERTY + " must be a positive multiple of 256; got: " + configured);
    }
    return value;
  }

  private static int resolvePositive(String property, int defaultValue) {
    String configured = System.getProperty(property);
    if (configured == null || configured.isBlank()) {
      return defaultValue;
    }
    int value = Integer.parseInt(configured.trim());
    if (value < 1) {
      throw new IllegalArgumentException("-D" + property + " must be positive: " + configured);
    }
    return value;
  }

  private static int ceilDiv(int a, int b) {
    return (a + b - 1) / b;
  }

  private static int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
  }
}
