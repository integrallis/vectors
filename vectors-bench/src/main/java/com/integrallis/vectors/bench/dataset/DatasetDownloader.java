package com.integrallis.vectors.bench.dataset;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Auto-downloads standard ANN-benchmark datasets from public mirrors.
 *
 * <p>Downloaded files are cached in the directory returned by {@link DatasetRegistry#dataDir()}.
 * Subsequent calls to {@link #ensureAvailable(String)} return immediately if the file already
 * exists.
 *
 * <p>Supported datasets (HDF5 format from ann-benchmarks.com):
 *
 * <ul>
 *   <li>{@code sift-128-euclidean} — 1M × 128-dim, Euclidean (~500 MB)
 *   <li>{@code glove-25-angular} — 1.2M × 25-dim, Angular (~120 MB)
 *   <li>{@code glove-50-angular} — 1.2M × 50-dim, Angular (~230 MB)
 *   <li>{@code glove-100-angular} — 1.2M × 100-dim, Angular (~460 MB)
 *   <li>{@code glove-200-angular} — 1.2M × 200-dim, Angular (~920 MB)
 *   <li>{@code fashion-mnist-784-euclidean} — 60K × 784-dim, Euclidean (~30 MB)
 *   <li>{@code mnist-784-euclidean} — 60K × 784-dim, Euclidean (~30 MB)
 *   <li>{@code nytimes-256-angular} — 290K × 256-dim, Angular (~300 MB)
 *   <li>{@code lastfm-64-dot} — 292K × 64-dim, Dot (~75 MB)
 *   <li>{@code gist-960-euclidean} — 1M × 960-dim, Euclidean (~3.6 GB)
 *   <li>{@code deep-image-96-angular} — 10M × 96-dim, Angular (~3.6 GB)
 * </ul>
 *
 * <p>Usage:
 *
 * <pre>{@code
 * Path sift = DatasetDownloader.ensureAvailable("sift-128-euclidean");
 * float[][] train = Hdf5Loader.readTrainVectors(sift);
 * }</pre>
 */
public final class DatasetDownloader {

  /**
   * Primary URL for ANN-Benchmarks HDF5 datasets (HTTPS, HuggingFace CDN). Falls back to the
   * original ann-benchmarks.com mirror if the primary fails.
   */
  private static final String ANN_BENCH_PRIMARY_URL =
      "https://huggingface.co/datasets/erikbern/ann-benchmarks/resolve/main/data/";

  /** Legacy mirror — kept as fallback. */
  private static final String ANN_BENCH_FALLBACK_URL = "https://ann-benchmarks.com/";

  /**
   * Known dataset names. SHA-256 verification is skipped for all datasets because the
   * HuggingFace-hosted HDF5 files are periodically rebuilt by the ANN-Benchmarks maintainers —
   * pinning a specific hash would cause spurious failures after a rebuild. Integrity is guaranteed
   * by HTTPS transport instead.
   */
  private static final Set<String> KNOWN_DATASETS =
      Set.of(
          "sift-128-euclidean",
          "glove-25-angular",
          "glove-50-angular",
          "glove-100-angular",
          "glove-200-angular",
          "fashion-mnist-784-euclidean",
          "mnist-784-euclidean",
          "nytimes-256-angular",
          "lastfm-64-dot",
          "gist-960-euclidean",
          "deep-image-96-angular");

  /**
   * Checksums map kept for API compatibility; all values are {@code null} (no verification). Backed
   * by a {@link HashMap} to support null values.
   */
  private static final Map<String, String> CHECKSUMS;

  static {
    CHECKSUMS = new HashMap<>();
    for (String name : KNOWN_DATASETS) {
      CHECKSUMS.put(name, null);
    }
  }

  private DatasetDownloader() {}

  /**
   * Ensures the named dataset is available locally, downloading it if necessary.
   *
   * <p>The file is cached under {@code DatasetRegistry.annBenchDir()} as {@code name.hdf5}. If the
   * file already exists, this method returns immediately without network access.
   *
   * @param name the dataset name (e.g., {@code "sift-128-euclidean"})
   * @return path to the local HDF5 file
   * @throws IllegalArgumentException if the dataset name is not recognized
   * @throws IOException if the download or filesystem operation fails
   */
  public static Path ensureAvailable(String name) throws IOException {
    Objects.requireNonNull(name, "name must not be null");
    if (!CHECKSUMS.containsKey(name)) {
      throw new IllegalArgumentException(
          "Unknown dataset: '"
              + name
              + "'. Known datasets: "
              + String.join(", ", CHECKSUMS.keySet()));
    }

    Path target = DatasetRegistry.annBenchDir().resolve(name + ".hdf5");
    if (Files.exists(target)) {
      return target;
    }

    // Try HuggingFace CDN first, fall back to legacy ann-benchmarks.com mirror.
    String primaryUrl = ANN_BENCH_PRIMARY_URL + name + ".hdf5";
    try {
      download(primaryUrl, target, name);
    } catch (IOException e) {
      System.out.printf(
          "[DatasetDownloader] Primary URL failed (%s); retrying via fallback mirror...%n",
          e.getMessage());
      String fallbackUrl = ANN_BENCH_FALLBACK_URL + name + ".hdf5";
      download(fallbackUrl, target, name);
    }
    return target;
  }

  /**
   * Returns {@code true} if the named dataset is recognized (i.e., can be downloaded).
   *
   * @param name the dataset name
   * @return true if known
   */
  public static boolean isKnownDataset(String name) {
    return CHECKSUMS.containsKey(name);
  }

  // -------------------------------------------------------------------------
  // Internal
  // -------------------------------------------------------------------------

  private static void download(String url, Path target, String name) throws IOException {
    Files.createDirectories(target.getParent());
    Path tmp = target.resolveSibling(target.getFileName() + ".downloading");

    System.out.printf("[DatasetDownloader] Downloading %s...%n", name);
    System.out.printf("[DatasetDownloader]   URL: %s%n", url);

    try (HttpClient client = HttpClient.newHttpClient()) {
      HttpRequest request =
          HttpRequest.newBuilder()
              .uri(URI.create(url))
              .header("User-Agent", "java-vectors-bench/1.0")
              .GET()
              .build();

      HttpResponse<InputStream> response =
          client.send(request, HttpResponse.BodyHandlers.ofInputStream());

      if (response.statusCode() != 200) {
        throw new IOException("HTTP " + response.statusCode() + " downloading " + url);
      }

      long contentLength = response.headers().firstValueAsLong("Content-Length").orElse(-1);

      MessageDigest digest = newSha256();
      long written = 0;
      long lastReport = 0;

      try (InputStream in = response.body();
          OutputStream out = Files.newOutputStream(tmp)) {
        byte[] buf = new byte[8192];
        int n;
        while ((n = in.read(buf)) > 0) {
          out.write(buf, 0, n);
          digest.update(buf, 0, n);
          written += n;
          if (written - lastReport > 50_000_000L) {
            lastReport = written;
            if (contentLength > 0) {
              System.out.printf(
                  "[DatasetDownloader]   Progress: %.1f MB / %.1f MB (%.0f%%)%n",
                  written / 1e6, contentLength / 1e6, 100.0 * written / contentLength);
            } else {
              System.out.printf("[DatasetDownloader]   Progress: %.1f MB%n", written / 1e6);
            }
          }
        }
      }

      System.out.printf("[DatasetDownloader]   Downloaded %.1f MB%n", written / 1e6);

      // Verify checksum if known.
      String expectedHash = CHECKSUMS.get(name);
      if (expectedHash != null) {
        String actualHash = HexFormat.of().formatHex(digest.digest());
        if (!expectedHash.equalsIgnoreCase(actualHash)) {
          Files.deleteIfExists(tmp);
          throw new IOException(
              "SHA-256 mismatch for "
                  + name
                  + ": expected "
                  + expectedHash
                  + " but got "
                  + actualHash);
        }
        System.out.println("[DatasetDownloader]   SHA-256 verified.");
      }

      // Atomic move into place.
      Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE);
      System.out.printf("[DatasetDownloader]   Saved to %s%n", target);

    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException("Download interrupted for " + name, e);
    } finally {
      Files.deleteIfExists(tmp);
    }
  }

  private static MessageDigest newSha256() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException e) {
      throw new AssertionError("SHA-256 not available", e);
    }
  }
}
