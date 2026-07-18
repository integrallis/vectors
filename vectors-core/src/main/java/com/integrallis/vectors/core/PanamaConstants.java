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

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * Constants for Panama Vector API SIMD operations. Resolves platform capabilities deterministically
 * at class-load time to enable conditional FMA dispatch and species selection.
 *
 * <p>The conditional FMA policy follows Apache Lucene's conservative platform heuristics. It does
 * not benchmark at class-load time, so unrelated startup load cannot change arithmetic or model
 * output. System properties allow overriding the policy for measured deployments and testing.
 */
public final class PanamaConstants {

  private static final Path LINUX_CPUINFO = Path.of("/proc/cpuinfo");

  /**
   * Default ceiling on SIMD register width, in bits. Matches Lucene's {@code
   * MAX_BITS_PER_VECTOR=256}: on many Intel server parts a sustained AVX-512 (512-bit) workload
   * triggers a frequency downclock that makes the wider registers a net loss for the rest of the
   * process. Capping at 256 bits keeps the common case fast; opt back in with {@code
   * -Dvectors.maxBits=512} on hardware (e.g. recent Sapphire Rapids, AMD Zen 4+) where 512-bit does
   * not downclock.
   */
  static final int DEFAULT_MAX_BITS = 256;

  private static final int SMALLEST_SIMD_BITS = 64;
  private static final int LARGEST_SIMD_BITS = 512;

  /**
   * The resolved SIMD register-width ceiling in bits. Defaults to {@link #DEFAULT_MAX_BITS};
   * override with {@code -Dvectors.maxBits=<64|128|256|512>}. The effective species width is {@code
   * min(MAX_BITS, hardware-preferred)} — this never widens beyond what the platform prefers, it
   * only caps it.
   */
  public static final int MAX_BITS = resolveMaxBits();

  /**
   * The platform's hardware-preferred float SIMD width in bits, before any {@link #MAX_BITS} cap.
   */
  public static final int PREFERRED_BITS = FloatVector.SPECIES_PREFERRED.vectorBitSize();

  private PanamaConstants() {}

  /**
   * Returns {@code preferred} capped to {@link #MAX_BITS}. When the platform-preferred width is
   * already within the cap the preferred species is returned unchanged; otherwise the same element
   * type is returned at the capped shape. Because this only ever caps downward to a width the
   * platform already supports, the returned species is always natively executable.
   */
  static <E> VectorSpecies<E> preferredSpecies(VectorSpecies<E> preferred) {
    if (preferred.vectorBitSize() <= MAX_BITS) {
      return preferred;
    }
    return VectorSpecies.of(preferred.elementType(), VectorShape.forBitSize(MAX_BITS));
  }

  static int resolveMaxBits() {
    String override = System.getProperty("vectors.maxBits");
    if (override == null || override.isBlank()) {
      return DEFAULT_MAX_BITS;
    }
    return parseMaxBits(override.trim());
  }

  static int parseMaxBits(String value) {
    int parsed;
    try {
      parsed = Integer.parseInt(value);
    } catch (NumberFormatException e) {
      throw new IllegalArgumentException(
          "-Dvectors.maxBits must be an integer power of two in ["
              + SMALLEST_SIMD_BITS
              + ", "
              + LARGEST_SIMD_BITS
              + "]; got: "
              + value,
          e);
    }
    if (parsed < SMALLEST_SIMD_BITS || parsed > LARGEST_SIMD_BITS || (parsed & (parsed - 1)) != 0) {
      throw new IllegalArgumentException(
          "-Dvectors.maxBits must be a power of two in ["
              + SMALLEST_SIMD_BITS
              + ", "
              + LARGEST_SIMD_BITS
              + "]; got: "
              + parsed);
    }
    return parsed;
  }

  /**
   * Whether the platform has fast vector FMA. When false, SIMD code should use {@code
   * a.mul(b).add(c)} instead of {@code a.fma(b, c)}.
   *
   * <p>Can be overridden with {@code -Dvectors.useVectorFMA=true|false}.
   */
  public static final boolean HAS_FAST_VECTOR_FMA = detectFastVectorFma();

  /**
   * Whether the platform has fast scalar FMA ({@link Math#fma}).
   *
   * <p>Can be overridden with {@code -Dvectors.useScalarFMA=true|false}.
   */
  public static final boolean HAS_FAST_SCALAR_FMA = detectFastScalarFma();

  /**
   * Whether the platform reports ARM SVE (Scalable Vector Extension) support. Detected once at
   * class load by combining the JVM's {@code os.arch} (must be {@code aarch64}) with the {@code
   * /proc/cpuinfo} feature flag {@code sve}. On x86/Apple Silicon the result is always {@code
   * false}; that is the correct "no SVE path here" answer and existing fixed-species kernels are
   * unaffected.
   *
   * <p>This is detection-only scaffolding (P3.5): no kernel currently dispatches on it. A future
   * predicated/scalable-vector kernel can branch on {@code HAS_SVE} to enter the SVE path while
   * falling back to {@link #FLOAT_SPECIES} elsewhere. Can be forced via {@code
   * -Dvectors.forceSve=true|false} for testing.
   */
  public static final boolean HAS_SVE = detectSve();

  private static boolean detectFastVectorFma() {
    String override = System.getProperty("vectors.useVectorFMA");
    if (override != null) {
      return Boolean.parseBoolean(override);
    }
    return hasFastVectorFma(
        System.getProperty("os.arch", ""),
        System.getProperty("os.name", ""),
        readLinuxCpuInfo(),
        PREFERRED_BITS);
  }

  private static boolean detectFastScalarFma() {
    String override = System.getProperty("vectors.useScalarFMA");
    if (override != null) {
      return Boolean.parseBoolean(override);
    }
    // Math.fma is generally fast on all modern CPUs
    return true;
  }

  private static boolean detectSve() {
    String override = System.getProperty("vectors.forceSve");
    if (override != null) {
      return Boolean.parseBoolean(override);
    }
    String arch = System.getProperty("os.arch", "");
    if (!isArmArch(arch)) {
      return false;
    }
    return hasSveCpuFlag(readLinuxCpuInfo());
  }

  /** Returns true when {@code os.arch} identifies a 64-bit ARM (aarch64 / arm64). */
  static boolean isArmArch(String arch) {
    return switch (arch.toLowerCase(Locale.ROOT)) {
      case "aarch64", "arm64" -> true;
      default -> false;
    };
  }

  /**
   * Returns true if the linux cpuinfo carries the {@code sve} feature flag in its {@code Features:}
   * line. On macOS / Windows / containers without {@code /proc/cpuinfo} the empty CpuInfo returns
   * false (correct: we cannot prove SVE without explicit evidence).
   */
  static boolean hasSveCpuFlag(CpuInfo cpuInfo) {
    return cpuInfo.sve();
  }

  static boolean hasFastVectorFma(String arch, String osName, CpuInfo cpuInfo, int preferredBits) {
    // Lucene leaves vector FMA off on Apple Silicon and enables it on other AArch64 platforms.
    if (isArmArch(arch)) {
      return !osName.toLowerCase(Locale.ROOT).contains("mac");
    }
    if (!isX86Arch(arch)) {
      return false;
    }
    // Zen benefits from vector FMA at AVX2 width, while narrower AMD vectors remain conservative.
    if (cpuInfo.amd()) {
      return cpuInfo.fma() && preferredBits >= 256;
    }
    // Intel x86 has consistently fast vector FMA, including Intel-based Macs.
    return true;
  }

  static boolean isX86Arch(String arch) {
    return switch (arch.toLowerCase(Locale.ROOT)) {
      case "amd64", "x86_64", "x86", "i386", "i486", "i586", "i686" -> true;
      default -> false;
    };
  }

  static CpuInfo parseCpuInfo(String cpuInfo) {
    boolean amd = false;
    boolean fma = false;
    boolean sve = false;
    String[] lines = cpuInfo.split("\\R");
    for (String line : lines) {
      int sep = line.indexOf(':');
      if (sep < 0) {
        continue;
      }
      String key = line.substring(0, sep).trim().toLowerCase(Locale.ROOT);
      String value = line.substring(sep + 1).trim().toLowerCase(Locale.ROOT);
      if (key.equals("vendor_id") && value.equals("authenticamd")) {
        amd = true;
      } else if (key.equals("model name") && value.contains("amd")) {
        amd = true;
      } else if (key.equals("flags") || key.equals("features")) {
        if (hasCpuFlag(value, "fma")) fma = true;
        // The ARM Features: line uses the lowercase token "sve". Containers that report SVE2
        // typically still list sve as well, so checking the single token is sufficient.
        if (hasCpuFlag(value, "sve")) sve = true;
      }
    }
    return new CpuInfo(amd, fma, sve);
  }

  static boolean isAmdWithFma(CpuInfo cpuInfo) {
    return cpuInfo.amd() && cpuInfo.fma();
  }

  private static CpuInfo readLinuxCpuInfo() {
    if (!Files.exists(LINUX_CPUINFO)) {
      return CpuInfo.empty();
    }
    try {
      return parseCpuInfo(Files.readString(LINUX_CPUINFO));
    } catch (IOException e) {
      throw new UncheckedIOException("failed to read " + LINUX_CPUINFO, e);
    }
  }

  private static boolean hasCpuFlag(String flags, String expected) {
    String[] tokens = flags.split("\\s+");
    for (String token : tokens) {
      if (token.equals(expected)) {
        return true;
      }
    }
    return false;
  }

  record CpuInfo(boolean amd, boolean fma, boolean sve) {
    static CpuInfo empty() {
      return new CpuInfo(false, false, false);
    }
  }
}
