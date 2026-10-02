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

import java.util.Locale;
import java.util.concurrent.atomic.LongAdder;

/**
 * Arithmetic domain of the GGUF batched matmul entry points for Q4_K, Q6_K and Q8_0.
 *
 * <p>The process default is selected once with {@code -Dvectors.gguf.batchedMatmulKernel=integer|
 * band|dispatch} (default {@code integer}) and is reported by {@link VectorRuntimeCapabilities
 * #ggufBatchedMatmulKernel()}. Explicit overloads taking this enum bypass the property. The choice
 * applies to every Q4_K, Q6_K and Q8_0 batched matmul entry point that receives unquantized
 * activations: single-matrix, dual and triple (grouped) calls, and the Q6_K overload taking a
 * {@link GgufQ6BatchedKernel} (which selects the integer register tile when the integer arm runs).
 *
 * <p><b>Experimental.</b> {@link #BAND_F32} and {@link #BATCH_DISPATCH} are not numerically
 * identical to {@link #INTEGER} (see the constants' documentation) and are not defaults.
 *
 * <p><b>Observability.</b> Every routed matrix is counted per format and arm ({@link
 * #routingReport()}); {@code -Dvectors.gguf.batchedMatmulKernel.report=true} prints the report to
 * standard error at JVM exit, so a model-level run can prove which arm executed.
 */
public enum GgufBatchedMatmulKernel {
  /**
   * The established kernels: activations are quantized to Q8_K/Q8_0 in caller scratch and the dot
   * products run in integer lanes.
   */
  INTEGER,

  /**
   * Experimental dequantise-to-F32 band GEMM: a cache-sized panel of weight rows is dequantized to
   * F32 scratch and swept against the unquantized activations with a vector FMA register tile. The
   * caller's Q8 scratch arrays are validated but not written. Results equal the float dot product
   * of the dequantized weights with the activations, so they differ from {@link #INTEGER} by that
   * kernel's activation quantization error. Requires the Panama provider.
   */
  BAND_F32,

  /**
   * Batch-size dispatch: {@link #INTEGER} below a per-format activation-row threshold, {@link
   * #BAND_F32} at or above it. The thresholds are fixed to the values pre-registered and passed at
   * kernel level on AMD EPYC Genoa (512-bit and 256-bit species) on 2026-09-17: Q4_K 4, Q6_K 32,
   * Q8_0 4 (vectors-bench {@code jmh-results/2026-09-16-band-gemm}). In grouped calls each matrix
   * is routed by its own format. Requires the Panama provider.
   */
  BATCH_DISPATCH;

  static final int FORMAT_Q4_K = 0;
  static final int FORMAT_Q6_K = 1;
  static final int FORMAT_Q8_0 = 2;
  private static final String[] FORMAT_NAMES = {"Q4_K", "Q6_K", "Q8_0"};

  /** Pre-registered dispatch thresholds (smallest batch routed to the band arm), by format. */
  private static final int[] DISPATCH_MIN_BATCH = {4, 32, 4};

  /** System property printing {@link #routingReport()} to standard error at JVM exit. */
  public static final String REPORT_PROPERTY = "vectors.gguf.batchedMatmulKernel.report";

  /** {integer calls, integer rows, band calls, band rows} per format. */
  private static final LongAdder[] ROUTING = newAdders(FORMAT_NAMES.length * 4);

  /** System property selecting the process default. */
  public static final String PROPERTY = "vectors.gguf.batchedMatmulKernel";

  private static final GgufBatchedMatmulKernel CONFIGURED = parse(System.getProperty(PROPERTY));

  static {
    if (Boolean.getBoolean(REPORT_PROPERTY)) {
      Runtime.getRuntime()
          .addShutdownHook(
              new Thread(() -> System.err.println(routingReport()), "vectors-gguf-routing-report"));
    }
  }

  /** Returns the kernel requested by {@link #PROPERTY}, whether or not it can run here. */
  public static GgufBatchedMatmulKernel configured() {
    return CONFIGURED;
  }

  /**
   * Returns the kernel the default entry points actually run: the configured kernel, except that
   * {@link #BAND_F32} and {@link #BATCH_DISPATCH} degrade to {@link #INTEGER} when the Vector API
   * provider is not active (visible in {@link
   * VectorRuntimeCapabilities#ggufBatchedMatmulKernel()}).
   */
  public static GgufBatchedMatmulKernel active() {
    return CONFIGURED != INTEGER && !VectorizationProvider.isPanamaEnabled() ? INTEGER : CONFIGURED;
  }

  /** Whether this kernel runs the band arm for {@code batch} activation rows of {@code format}. */
  boolean usesBand(int format, int batch) {
    return switch (this) {
      case INTEGER -> false;
      case BAND_F32 -> true;
      case BATCH_DISPATCH -> batch >= DISPATCH_MIN_BATCH[format];
    };
  }

  /** Smallest activation-row count routed to the band arm by {@link #BATCH_DISPATCH}. */
  static int dispatchMinBatch(int format) {
    return DISPATCH_MIN_BATCH[format];
  }

  /** Counts one routed matrix call. */
  static void record(int format, boolean band, int batch) {
    int base = format * 4 + (band ? 2 : 0);
    ROUTING[base].increment();
    ROUTING[base + 1].add(batch);
  }

  /** {integer calls, integer rows, band calls, band rows} for {@code format}. */
  static long[] routingCounts(int format) {
    long[] counts = new long[4];
    for (int i = 0; i < 4; i++) {
      counts[i] = ROUTING[format * 4 + i].sum();
    }
    return counts;
  }

  static void resetRoutingCounts() {
    for (LongAdder adder : ROUTING) {
      adder.reset();
    }
  }

  /**
   * One line: the active mode and, per format, {@code integer=calls/rows band=calls/rows} counted
   * over every Q4_K/Q6_K/Q8_0 batched matmul matrix routed since start (or the last reset).
   */
  public static String routingReport() {
    StringBuilder report =
        new StringBuilder("vectors-gguf-batched-matmul-routing mode=").append(modeName(active()));
    for (int format = 0; format < FORMAT_NAMES.length; format++) {
      long[] counts = routingCounts(format);
      report
          .append(' ')
          .append(FORMAT_NAMES[format])
          .append(" integer=")
          .append(counts[0])
          .append('/')
          .append(counts[1])
          .append(" band=")
          .append(counts[2])
          .append('/')
          .append(counts[3]);
    }
    return report.toString();
  }

  private static String modeName(GgufBatchedMatmulKernel kernel) {
    return switch (kernel) {
      case INTEGER -> "integer";
      case BAND_F32 -> "band";
      case BATCH_DISPATCH -> "dispatch";
    };
  }

  private static LongAdder[] newAdders(int count) {
    LongAdder[] adders = new LongAdder[count];
    for (int i = 0; i < count; i++) {
      adders[i] = new LongAdder();
    }
    return adders;
  }

  static GgufBatchedMatmulKernel parse(String configured) {
    if (configured == null || configured.isBlank()) {
      return INTEGER;
    }
    return switch (configured.trim().toLowerCase(Locale.ROOT)) {
      case "integer", "int8" -> INTEGER;
      case "band", "band_f32", "band-f32" -> BAND_F32;
      case "dispatch", "batch_dispatch", "batch-dispatch" -> BATCH_DISPATCH;
      default ->
          throw new IllegalArgumentException(
              "-D" + PROPERTY + " must be integer, band or dispatch; got: " + configured);
    };
  }

  /**
   * Resolved configuration of the {@link #BAND_F32} arm (tile, column block, panel size, lanes)
   * whether or not it is the process default, so a caller selecting it explicitly can record what
   * ran; {@code "unavailable"} without the Vector API provider.
   */
  public static String bandConfiguration() {
    return VectorizationProvider.isPanamaEnabled() ? GgufBandGemm.describe() : "unavailable";
  }

  /**
   * The band arm's K-quant dequantisation arm, requested and effective ({@code
   * -Dvectors.gguf.band.dequant}); reported separately so the band configuration string is
   * unchanged.
   */
  public static String bandDequantConfiguration() {
    return VectorizationProvider.isPanamaEnabled() ? GgufKQuantDequant.describe() : "unavailable";
  }

  /** Dispatch description with its thresholds and the band arm's resolved configuration. */
  static String describeDispatch() {
    StringBuilder description = new StringBuilder("dispatch(band-at-batch>=");
    for (int format = 0; format < FORMAT_NAMES.length; format++) {
      description
          .append(format == 0 ? "" : ",")
          .append(FORMAT_NAMES[format])
          .append(':')
          .append(DISPATCH_MIN_BATCH[format]);
    }
    return description.append(";band=").append(bandConfiguration()).append(')').toString();
  }

  /** Human-readable description including the band arm's resolved tile, block and panel sizes. */
  static String describeActive() {
    return switch (active()) {
      case INTEGER ->
          CONFIGURED == INTEGER
              ? "integer"
              : "integer ("
                  + CONFIGURED.name().toLowerCase(Locale.ROOT)
                  + " requested; Vector API unavailable)";
      case BAND_F32 -> GgufBandGemm.describe();
      case BATCH_DISPATCH -> describeDispatch();
    };
  }
}
