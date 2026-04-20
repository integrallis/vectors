package com.integrallis.vectors.bench;

import com.integrallis.vectors.bench.report.BenchmarkReporter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Tracks which benchmark configurations have already been completed across restarts.
 *
 * <p>On startup the store reads an existing CSV of {@link
 * com.integrallis.vectors.bench.report.BenchmarkResult}s (if any) and builds two sets:
 *
 * <ul>
 *   <li>Per-row keys {@code (dataset|algorithm|buildParams|searchParams)} — used to decide whether
 *       a single measurement has already been recorded.
 *   <li>Per-build keys {@code (dataset|algorithm|buildParams)} — used with {@code .completed}
 *       marker files to decide whether an entire index-build + search sweep is done and may be
 *       skipped wholesale.
 * </ul>
 *
 * <p>The marker files live in a sibling {@code markers/} directory and contain no payload; their
 * presence is the signal. Marker writes and CSV appends are idempotent: reruns of a completed
 * configuration are no-ops.
 */
public final class CheckpointStore {

  private final Path csvFile;
  private final Path markersDir;
  private final Set<String> completedRowKeys = new HashSet<>();
  private final Set<String> completedBuildKeys = new HashSet<>();

  public CheckpointStore(Path csvFile, Path markersDir) {
    this.csvFile = csvFile;
    this.markersDir = markersDir;
    load();
  }

  private void load() {
    try {
      if (markersDir != null) Files.createDirectories(markersDir);
      if (csvFile != null && Files.exists(csvFile)) {
        List<String> lines = Files.readAllLines(csvFile, StandardCharsets.UTF_8);
        for (int i = 1; i < lines.size(); i++) { // skip header
          String line = lines.get(i);
          if (line.isBlank()) continue;
          // CSV columns: timestamp, dataset, algorithm, build_params, search_params, ...
          List<String> cols = parseCsvRow(line);
          if (cols.size() < 5) continue;
          String rowKey = rowKey(cols.get(1), cols.get(2), cols.get(3), cols.get(4));
          completedRowKeys.add(rowKey);
        }
      }
      if (markersDir != null && Files.isDirectory(markersDir)) {
        try (var stream = Files.list(markersDir)) {
          stream
              .filter(p -> p.getFileName().toString().endsWith(".completed"))
              .forEach(
                  p -> {
                    try {
                      // The marker file's content is the original (unsanitized) build key; we
                      // avoid using the filename because safeFileName is lossy.
                      String key = Files.readString(p, StandardCharsets.UTF_8).trim();
                      if (!key.isEmpty()) completedBuildKeys.add(key);
                    } catch (IOException ignored) {
                      // Ignore unreadable marker files — they'll be re-created on completion.
                    }
                  });
        }
      }
    } catch (IOException e) {
      throw new UncheckedIOException("Failed to load checkpoint state from " + csvFile, e);
    }
  }

  /** Returns true iff the entire (dataset, algorithm, buildParams) build sweep is already done. */
  public boolean isBuildCompleted(
      String dataset, String algorithm, Map<String, String> buildParams) {
    return completedBuildKeys.contains(buildKey(dataset, algorithm, buildParams));
  }

  /** Returns true iff a single measurement row has already been recorded. */
  public boolean isRowCompleted(
      String dataset,
      String algorithm,
      Map<String, String> buildParams,
      Map<String, String> searchParams) {
    return completedRowKeys.contains(
        rowKey(
            dataset,
            algorithm,
            BenchmarkReporter.formatParams(buildParams),
            BenchmarkReporter.formatParams(searchParams)));
  }

  /** Marks a build sweep as completed by writing a {@code .completed} marker file. */
  public void markBuildCompleted(
      String dataset, String algorithm, Map<String, String> buildParams) {
    String key = buildKey(dataset, algorithm, buildParams);
    completedBuildKeys.add(key);
    if (markersDir == null) return;
    try {
      Files.createDirectories(markersDir);
      Path marker = markersDir.resolve(safeFileName(key) + ".completed");
      Files.writeString(marker, key, StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new UncheckedIOException("Failed to write completion marker for " + key, e);
    }
  }

  /** Records that a CSV row has been written (in-memory; CSV append happens elsewhere). */
  public void markRowCompleted(
      String dataset,
      String algorithm,
      Map<String, String> buildParams,
      Map<String, String> searchParams) {
    completedRowKeys.add(
        rowKey(
            dataset,
            algorithm,
            BenchmarkReporter.formatParams(buildParams),
            BenchmarkReporter.formatParams(searchParams)));
  }

  private static String buildKey(
      String dataset, String algorithm, Map<String, String> buildParams) {
    return dataset + "|" + algorithm + "|" + BenchmarkReporter.formatParams(buildParams);
  }

  private static String rowKey(
      String dataset, String algorithm, String buildParamsStr, String searchParamsStr) {
    return dataset + "|" + algorithm + "|" + buildParamsStr + "|" + searchParamsStr;
  }

  private static String safeFileName(String s) {
    return s.replaceAll("[^A-Za-z0-9._-]+", "_");
  }

  /** Minimal CSV row parser that handles quoted fields containing commas. */
  static List<String> parseCsvRow(String line) {
    List<String> out = new java.util.ArrayList<>();
    StringBuilder cur = new StringBuilder();
    boolean inQuotes = false;
    for (int i = 0; i < line.length(); i++) {
      char c = line.charAt(i);
      if (inQuotes) {
        if (c == '"') {
          if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
            cur.append('"');
            i++;
          } else inQuotes = false;
        } else cur.append(c);
      } else {
        if (c == ',') {
          out.add(cur.toString());
          cur.setLength(0);
        } else if (c == '"') inQuotes = true;
        else cur.append(c);
      }
    }
    out.add(cur.toString());
    return out;
  }
}
