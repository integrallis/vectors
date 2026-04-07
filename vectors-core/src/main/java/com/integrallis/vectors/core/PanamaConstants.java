package com.integrallis.vectors.core;

/**
 * Constants for Panama Vector API SIMD operations. Detects platform capabilities at class-load time
 * to enable conditional FMA dispatch and species selection.
 *
 * <p>The conditional FMA pattern comes from Apache Lucene, which discovered that FMA is slower on
 * some AMD CPUs (certain Zen architectures). System properties allow overriding for testing.
 */
public final class PanamaConstants {

  private PanamaConstants() {}

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

  private static boolean detectFastVectorFma() {
    String override = System.getProperty("vectors.useVectorFMA");
    if (override != null) {
      return Boolean.parseBoolean(override);
    }
    // FMA is fast on Intel (Haswell+) and AMD (Zen 4+/EPYC), but can be slower
    // on some AMD Zen 1-3 processors for vector operations. Default to true since
    // JDK 25 users likely have modern hardware; allow override for problematic CPUs.
    return !isAmdCpuWithSlowFma();
  }

  private static boolean detectFastScalarFma() {
    String override = System.getProperty("vectors.useScalarFMA");
    if (override != null) {
      return Boolean.parseBoolean(override);
    }
    // Math.fma is generally fast on all modern CPUs
    return true;
  }

  private static boolean isAmdCpuWithSlowFma() {
    // On x86, check CPU vendor. On ARM (Apple Silicon, Graviton), FMA is always fast.
    String arch = System.getProperty("os.arch", "");
    if (!arch.equals("amd64") && !arch.equals("x86_64")) {
      return false; // ARM/other: FMA is fast
    }
    // Check for AMD CPU with known slow vector FMA
    // Lucene uses a more sophisticated detection via disassembly; we use a simpler heuristic.
    // AMD Zen 1-3 can have slower vector FMA than mul+add on 256-bit operations.
    // For now, trust FMA on all platforms (user can override).
    return false;
  }
}
