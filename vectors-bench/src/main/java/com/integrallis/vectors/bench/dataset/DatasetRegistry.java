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
 *   <li>{@code <research>/data/}, where {@code <research>} is found by {@link #researchDir()}:
 *       {@code VECTORS_RESEARCH_DIR} if set, else the nearest {@code research/} directory located
 *       by walking up from the working directory.
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

  /**
   * Fallback: SIFT Small ships inside the cloned {@code jvector} sub-repo, under the research tree.
   *
   * <p>Resolved by walking up from the working directory rather than by a fixed relative path. A
   * hard-coded {@code ../../research/...} is silently wrong whenever the research tree does not sit
   * exactly two levels up, and this fallback does not fail loudly when it is wrong -- it makes
   * {@link #isSiftSmallAvailable()} return {@code false}, which disables every {@code @EnabledIf}
   * test gated on it. Those tests then report as not-run rather than as failing, so the breakage is
   * invisible. {@link #researchDir()} is the single place that decision is made.
   */
  private static final Path SIFT_SMALL_BUILTIN = researchDir().resolve("repos/jvector/siftsmall");

  private DatasetRegistry() {}

  // -------------------------------------------------------------------------
  // Research tree discovery
  // -------------------------------------------------------------------------

  /** How far up the directory chain to look for the research tree before giving up. */
  private static final int RESEARCH_SEARCH_DEPTH = 8;

  /**
   * Locates the {@code research/} tree holding downloaded datasets and cloned reference
   * implementations.
   *
   * <p>Honors {@code VECTORS_RESEARCH_DIR} when set. Otherwise walks up from the working directory
   * looking for a directory named {@code research} that contains {@code data} or {@code repos},
   * which makes the harness resolve the same way whether it is launched from a subproject, the
   * Gradle root, or a git worktree. Falls back to {@code ../../research} so the return is never
   * null; callers probe for existence.
   */
  public static Path researchDir() {
    String env = System.getenv("VECTORS_RESEARCH_DIR");
    if (env != null && !env.isBlank()) {
      return Path.of(env);
    }
    Path cursor = Path.of("").toAbsolutePath();
    for (int i = 0; i < RESEARCH_SEARCH_DEPTH && cursor != null; i++) {
      Path candidate = cursor.resolve("research");
      if (Files.isDirectory(candidate.resolve("repos"))
          || Files.isDirectory(candidate.resolve("data"))) {
        return candidate;
      }
      cursor = cursor.getParent();
    }
    return Path.of("../../research");
  }

  // -------------------------------------------------------------------------
  // Root data directory
  // -------------------------------------------------------------------------

  /**
   * Returns the root data directory: {@code VECTORS_BENCH_DATA} when set, otherwise {@code data/}
   * inside the tree located by {@link #researchDir()}.
   */
  public static Path dataDir() {
    String env = System.getenv("VECTORS_BENCH_DATA");
    return env != null ? Path.of(env) : researchDir().resolve("data");
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

  // -------------------------------------------------------------------------
  // Cohere Wikipedia (768 dims, big-ann competition format)
  // -------------------------------------------------------------------------

  /**
   * Directory holding the Cohere Wikipedia set from big-ann-benchmarks.
   *
   * <p>This is the only published million-scale corpus we have found whose dimensionality sits in
   * the 512-768 band we actually ship against: SIFT is 128, GloVe is 100, GIST is 960, and the
   * OpenAI-embedded sets are 1536. Measuring graph behaviour at 100 or 960 dimensions and assuming
   * it carries to 768 is the kind of transfer this project does not grant.
   */
  public static Path wikipediaCohereDir() {
    return dataDir().resolve("wikipedia-cohere-768");
  }

  /** The 1M crop of the Cohere Wikipedia base vectors. */
  public static Path wikipediaCohereBase1M() {
    return wikipediaCohereDir().resolve("wikipedia_base_1M.bin");
  }

  /** The 5,000 Cohere Wikipedia query vectors. */
  public static Path wikipediaCohereQueries() {
    return wikipediaCohereDir().resolve("wikipedia_query.bin");
  }

  /** Published ground truth for the 1M crop (5,000 queries x 100 neighbors). */
  public static Path wikipediaCohereGroundTruth1M() {
    return wikipediaCohereDir().resolve("wikipedia-1M.gt");
  }

  /** Returns {@code true} if all three Cohere Wikipedia 1M files are present. */
  public static boolean isWikipediaCohere1MAvailable() {
    return Files.exists(wikipediaCohereBase1M())
        && Files.exists(wikipediaCohereQueries())
        && Files.exists(wikipediaCohereGroundTruth1M());
  }

  /** JUnit 5 {@code @EnabledIf} condition for the Cohere Wikipedia 1M set. */
  public static boolean wikipediaCohere1MAvailable() {
    return isWikipediaCohere1MAvailable();
  }

  /**
   * JUnit 5 {@code @EnabledIf} condition for just the query and ground-truth files, which are a few
   * megabytes and land long before the multi-gigabyte base crop. Lets the format reader be verified
   * against published bytes without waiting on the full download.
   */
  public static boolean wikipediaCohereQueriesAvailable() {
    return Files.exists(wikipediaCohereQueries()) && Files.exists(wikipediaCohereGroundTruth1M());
  }
}
