package com.integrallis.vectors.bench.dataset;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Resolves standard ANN-benchmark dataset paths at runtime.
 *
 * <p>Path resolution order:
 *
 * <ol>
 *   <li>The directory named by the {@code VECTORS_BENCH_DATA} environment variable, if set.
 *   <li>The sibling {@code research/data/} directory relative to the Gradle project root (two
 *       levels above the {@code vectors-bench} subproject working directory).
 *   <li>Dataset-specific fallbacks for datasets already present in the repository (e.g., SIFT Small
 *       is bundled inside {@code research/repos/jvector/siftsmall/}).
 * </ol>
 *
 * <p>All {@code isXxxAvailable()} methods perform a cheap {@link Files#exists} probe. They are
 * designed to be used as JUnit 5 {@code @EnabledIf} condition methods.
 */
public final class DatasetRegistry {

  /** Expected file names inside the SIFT Small directory (TexMex format). */
  private static final String SIFT_SMALL_BASE = "siftsmall_base.fvecs";

  private static final String SIFT_SMALL_QUERY = "siftsmall_query.fvecs";
  private static final String SIFT_SMALL_GT = "siftsmall_groundtruth.ivecs";

  /** Fallback: SIFT Small is bundled with the jvector sub-repo already cloned at this path. */
  private static final Path SIFT_SMALL_BUILTIN = Path.of("../../research/repos/jvector/siftsmall");

  private DatasetRegistry() {}

  // -------------------------------------------------------------------------
  // Root data directory
  // -------------------------------------------------------------------------

  /**
   * Returns the root data directory, honoring the {@code VECTORS_BENCH_DATA} environment variable
   * if set, otherwise defaulting to {@code ../../research/data/} relative to the working directory
   * of the {@code vectors-bench} subproject.
   */
  public static Path dataDir() {
    String env = System.getenv("VECTORS_BENCH_DATA");
    return env != null ? Path.of(env) : Path.of("../../research/data");
  }

  // -------------------------------------------------------------------------
  // SIFT Small (10K base, 100 queries, 128 dims, L2)
  // -------------------------------------------------------------------------

  /**
   * Returns the directory containing SIFT Small {@code .fvecs} / {@code .ivecs} files. Checks
   * {@code $VECTORS_BENCH_DATA/siftsmall/} first, then the bundled fallback location.
   */
  public static Path siftSmallDir() {
    Path dedicated = dataDir().resolve("siftsmall");
    return Files.isDirectory(dedicated) ? dedicated : SIFT_SMALL_BUILTIN;
  }

  /** Returns {@code true} if the SIFT Small dataset files are present and readable. */
  public static boolean isSiftSmallAvailable() {
    Path dir = siftSmallDir();
    return Files.exists(dir.resolve(SIFT_SMALL_BASE))
        && Files.exists(dir.resolve(SIFT_SMALL_QUERY))
        && Files.exists(dir.resolve(SIFT_SMALL_GT));
  }

  /**
   * Convenience method for use as a JUnit 5 {@code @EnabledIf} condition. Returns {@code true} when
   * SIFT Small is available.
   *
   * <pre>{@code
   * @EnabledIf("com.integrallis.vectors.bench.dataset.DatasetRegistry#isSiftSmallAvailable")
   * }</pre>
   */
  public static boolean siftSmallAvailable() {
    return isSiftSmallAvailable();
  }

  // -------------------------------------------------------------------------
  // SIFT 1M (1M base, 10K queries, 128 dims, L2)
  // -------------------------------------------------------------------------

  /** Returns the directory expected to contain SIFT 1M {@code .fvecs} / {@code .ivecs} files. */
  public static Path sift1MDir() {
    return dataDir().resolve("sift");
  }

  /** Returns {@code true} if the SIFT 1M dataset base file is present. */
  public static boolean isSift1MAvailable() {
    return Files.exists(sift1MDir().resolve("sift_base.fvecs"));
  }

  /** JUnit 5 {@code @EnabledIf} condition for SIFT 1M. */
  public static boolean sift1MAvailable() {
    return isSift1MAvailable();
  }

  // -------------------------------------------------------------------------
  // GIST 1M (1M base, 1K queries, 960 dims, L2)
  // -------------------------------------------------------------------------

  /** Returns the directory expected to contain GIST 1M {@code .fvecs} / {@code .ivecs} files. */
  public static Path gist1MDir() {
    return dataDir().resolve("gist");
  }

  /** Returns {@code true} if the GIST 1M dataset base file is present. */
  public static boolean isGist1MAvailable() {
    return Files.exists(gist1MDir().resolve("gist_base.fvecs"));
  }

  /** JUnit 5 {@code @EnabledIf} condition for GIST 1M. */
  public static boolean gist1MAvailable() {
    return isGist1MAvailable();
  }

  // -------------------------------------------------------------------------
  // ANN-Benchmarks HDF5 datasets (GloVe, NYTimes, etc.)
  // -------------------------------------------------------------------------

  /** Returns the directory expected to contain ANN-Benchmarks {@code .hdf5} files. */
  public static Path annBenchDir() {
    return dataDir().resolve("ann-benchmarks");
  }

  /** Returns the {@code .hdf5} file for a named ANN-Benchmarks dataset. */
  public static Path annBenchDataset(String name) {
    return annBenchDir().resolve(name + ".hdf5");
  }

  /** Returns {@code true} if the named ANN-Benchmarks HDF5 file is present. */
  public static boolean isAnnBenchAvailable(String name) {
    return Files.exists(annBenchDataset(name));
  }

  /** JUnit 5 {@code @EnabledIf} condition for the {@code glove-100-angular} dataset. */
  public static boolean glove100Available() {
    return isAnnBenchAvailable("glove-100-angular");
  }

  /** JUnit 5 {@code @EnabledIf} condition for the {@code nytimes-256-angular} dataset. */
  public static boolean nytimesAvailable() {
    return isAnnBenchAvailable("nytimes-256-angular");
  }
}
