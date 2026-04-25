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
package com.integrallis.vectors.bench.report;

import java.io.IOException;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Multi-format reporter for benchmark results.
 *
 * <p>Supports three output formats:
 *
 * <ul>
 *   <li><b>Console</b> — JVector-style tabular output with box-drawing characters
 *   <li><b>CSV</b> — machine-readable, one row per measurement point
 *   <li><b>JSON</b> — structured output suitable for regression tracking
 * </ul>
 *
 * <p>Usage:
 *
 * <pre>{@code
 * List<BenchmarkResult> results = ...;
 * BenchmarkReporter.console(results, System.out);
 * BenchmarkReporter.csv(results, Path.of("results.csv"));
 * BenchmarkReporter.json(results, Path.of("results.json"));
 * }</pre>
 */
public final class BenchmarkReporter {

  private BenchmarkReporter() {}

  // -------------------------------------------------------------------------
  // Console (box-drawing tables)
  // -------------------------------------------------------------------------

  /**
   * Prints benchmark results as formatted console tables, grouped by (dataset, algorithm,
   * buildParams).
   */
  public static void console(List<BenchmarkResult> results, PrintStream out) {
    Objects.requireNonNull(results);
    Objects.requireNonNull(out);

    // Group by (dataset, algorithm, buildParams) → list of search-param variants.
    Map<String, List<BenchmarkResult>> groups = new LinkedHashMap<>();
    for (BenchmarkResult r : results) {
      String key = r.dataset() + "|" + r.algorithm() + "|" + formatParams(r.buildParams());
      groups.computeIfAbsent(key, k -> new ArrayList<>()).add(r);
    }

    for (var entry : groups.entrySet()) {
      List<BenchmarkResult> group = entry.getValue();
      BenchmarkResult first = group.getFirst();

      // Header
      String title =
          first.algorithm().toUpperCase(Locale.ROOT)
              + " "
              + formatParams(first.buildParams())
              + " -- "
              + first.dataset();
      out.println();
      out.printf("%-80s%n", "\u250C" + "\u2500".repeat(78) + "\u2510");
      out.printf("\u2502 %-77s\u2502%n", title);

      // Column headers
      boolean hasRecall100 = group.stream().anyMatch(r -> r.recall100() >= 0);
      boolean hasCompression = group.stream().anyMatch(r -> r.compressionRatio() > 1.001);

      out.printf("\u251C%s\u2524%n", "\u2500".repeat(78));
      StringBuilder header = new StringBuilder();
      header.append(String.format("\u2502 %-18s", "SearchParams"));
      header.append(String.format("\u2502 %9s", "Recall@10"));
      if (hasRecall100) {
        header.append(String.format("\u2502 %10s", "Recall@100"));
      }
      header.append(String.format("\u2502 %8s", "QPS"));
      header.append(String.format("\u2502 %8s", "p50(us)"));
      header.append(String.format("\u2502 %8s", "p99(us)"));
      if (hasCompression) {
        header.append(String.format("\u2502 %6s", "Compr"));
      }
      header.append(String.format("\u2502 %8s", "Build(s)"));
      header.append("\u2502");
      out.println(header);

      out.printf("\u251C%s\u2524%n", "\u2500".repeat(78));

      // Rows (sorted by recall@10 ascending)
      group.sort(Comparator.comparingDouble(BenchmarkResult::recall10));
      for (BenchmarkResult r : group) {
        StringBuilder row = new StringBuilder();
        row.append(String.format("\u2502 %-18s", formatParams(r.searchParams())));
        row.append(String.format("\u2502 %9.3f", r.recall10()));
        if (hasRecall100) {
          row.append(
              r.recall100() >= 0
                  ? String.format("\u2502 %10.3f", r.recall100())
                  : String.format("\u2502 %10s", "-"));
        }
        row.append(String.format("\u2502 %,8.0f", r.qps()));
        row.append(String.format("\u2502 %,8.0f", r.p50Us()));
        row.append(String.format("\u2502 %,8.0f", r.p99Us()));
        if (hasCompression) {
          row.append(String.format("\u2502 %5.1fx", r.compressionRatio()));
        }
        row.append(String.format("\u2502 %8.1f", r.buildTimeSeconds()));
        row.append("\u2502");
        out.println(row);
      }

      out.printf("\u2514%s\u2518%n", "\u2500".repeat(78));
    }
  }

  // -------------------------------------------------------------------------
  // CSV
  // -------------------------------------------------------------------------

  private static final String CSV_HEADER =
      "timestamp,dataset,algorithm,build_params,search_params,"
          + "recall_10,recall_100,qps,p50_us,p95_us,p99_us,"
          + "build_time_s,index_size_mb,compression_ratio";

  /**
   * Writes benchmark results in CSV format, one row per measurement point. Appends to the file if
   * it already exists; writes the header only if the file is new.
   */
  public static void csv(List<BenchmarkResult> results, Path file) {
    Objects.requireNonNull(results);
    Objects.requireNonNull(file);

    try {
      Files.createDirectories(file.getParent());
      boolean writeHeader = !Files.exists(file) || Files.size(file) == 0;

      StringBuilder sb = new StringBuilder();
      if (writeHeader) {
        sb.append(CSV_HEADER).append('\n');
      }

      String ts = Instant.now().toString();
      for (BenchmarkResult r : results) {
        appendCsvRow(sb, r, ts);
      }

      Files.writeString(
          file,
          sb.toString(),
          StandardCharsets.UTF_8,
          StandardOpenOption.CREATE,
          StandardOpenOption.APPEND);
    } catch (IOException e) {
      throw new UncheckedIOException("Failed to write CSV to " + file, e);
    }
  }

  /**
   * Appends a single benchmark result to the CSV file, writing the header if the file is new or
   * empty. Each call flushes to disk so partial progress survives a crash or kill.
   */
  public static void csvAppend(BenchmarkResult result, Path file) {
    Objects.requireNonNull(result);
    Objects.requireNonNull(file);

    try {
      Files.createDirectories(file.getParent());
      boolean writeHeader = !Files.exists(file) || Files.size(file) == 0;

      StringBuilder sb = new StringBuilder();
      if (writeHeader) {
        sb.append(CSV_HEADER).append('\n');
      }
      appendCsvRow(sb, result, Instant.now().toString());

      Files.writeString(
          file,
          sb.toString(),
          StandardCharsets.UTF_8,
          StandardOpenOption.CREATE,
          StandardOpenOption.APPEND,
          StandardOpenOption.SYNC);
    } catch (IOException e) {
      throw new UncheckedIOException("Failed to append CSV row to " + file, e);
    }
  }

  private static void appendCsvRow(StringBuilder sb, BenchmarkResult r, String ts) {
    sb.append(ts).append(',');
    sb.append(csvEscape(r.dataset())).append(',');
    sb.append(csvEscape(r.algorithm())).append(',');
    sb.append(csvEscape(formatParams(r.buildParams()))).append(',');
    sb.append(csvEscape(formatParams(r.searchParams()))).append(',');
    sb.append(String.format(Locale.ROOT, "%.4f", r.recall10())).append(',');
    sb.append(r.recall100() >= 0 ? String.format(Locale.ROOT, "%.4f", r.recall100()) : "")
        .append(',');
    sb.append(String.format(Locale.ROOT, "%.1f", r.qps())).append(',');
    sb.append(String.format(Locale.ROOT, "%.1f", r.p50Us())).append(',');
    sb.append(String.format(Locale.ROOT, "%.1f", r.p95Us())).append(',');
    sb.append(String.format(Locale.ROOT, "%.1f", r.p99Us())).append(',');
    sb.append(String.format(Locale.ROOT, "%.2f", r.buildTimeSeconds())).append(',');
    sb.append(String.format(Locale.ROOT, "%.2f", r.indexSizeMb())).append(',');
    sb.append(String.format(Locale.ROOT, "%.2f", r.compressionRatio()));
    sb.append('\n');
  }

  // -------------------------------------------------------------------------
  // JSON
  // -------------------------------------------------------------------------

  /**
   * Writes benchmark results in JSON format. Overwrites the file if it already exists.
   *
   * <p>Output structure:
   *
   * <pre>{@code
   * {
   *   "timestamp": "...",
   *   "system": { "java": "...", "os": "...", "arch": "...", "processors": N },
   *   "results": [ ... ]
   * }
   * }</pre>
   */
  public static void json(List<BenchmarkResult> results, Path file) {
    Objects.requireNonNull(results);
    Objects.requireNonNull(file);

    try {
      Files.createDirectories(file.getParent());

      StringBuilder sb = new StringBuilder();
      sb.append("{\n");
      sb.append("  \"timestamp\": \"").append(Instant.now()).append("\",\n");
      sb.append("  \"system\": {\n");
      sb.append("    \"java\": \"")
          .append(jsonEscape(System.getProperty("java.version", "unknown")))
          .append("\",\n");
      sb.append("    \"os\": \"")
          .append(jsonEscape(System.getProperty("os.name", "unknown")))
          .append("\",\n");
      sb.append("    \"arch\": \"")
          .append(jsonEscape(System.getProperty("os.arch", "unknown")))
          .append("\",\n");
      sb.append("    \"processors\": ")
          .append(Runtime.getRuntime().availableProcessors())
          .append("\n");
      sb.append("  },\n");
      sb.append("  \"results\": [\n");

      for (int i = 0; i < results.size(); i++) {
        BenchmarkResult r = results.get(i);
        sb.append("    {\n");
        sb.append("      \"dataset\": \"").append(jsonEscape(r.dataset())).append("\",\n");
        sb.append("      \"algorithm\": \"").append(jsonEscape(r.algorithm())).append("\",\n");
        sb.append("      \"build_params\": ").append(mapToJson(r.buildParams())).append(",\n");
        sb.append("      \"search_params\": ").append(mapToJson(r.searchParams())).append(",\n");
        sb.append("      \"metrics\": {\n");
        sb.append(String.format(Locale.ROOT, "        \"recall_10\": %.4f,\n", r.recall10()));
        if (r.recall100() >= 0) {
          sb.append(String.format(Locale.ROOT, "        \"recall_100\": %.4f,\n", r.recall100()));
        }
        sb.append(String.format(Locale.ROOT, "        \"qps\": %.1f,\n", r.qps()));
        sb.append(String.format(Locale.ROOT, "        \"p50_us\": %.1f,\n", r.p50Us()));
        sb.append(String.format(Locale.ROOT, "        \"p95_us\": %.1f,\n", r.p95Us()));
        sb.append(String.format(Locale.ROOT, "        \"p99_us\": %.1f,\n", r.p99Us()));
        sb.append(
            String.format(Locale.ROOT, "        \"build_time_s\": %.2f,\n", r.buildTimeSeconds()));
        sb.append(
            String.format(Locale.ROOT, "        \"index_size_mb\": %.2f,\n", r.indexSizeMb()));
        sb.append(
            String.format(
                Locale.ROOT, "        \"compression_ratio\": %.2f\n", r.compressionRatio()));
        sb.append("      }");
        if (!r.extra().isEmpty()) {
          sb.append(",\n      \"extra\": ").append(mapToJson(r.extra()));
        }
        sb.append("\n    }");
        if (i < results.size() - 1) sb.append(',');
        sb.append('\n');
      }

      sb.append("  ]\n");
      sb.append("}\n");

      Files.writeString(file, sb.toString(), StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new UncheckedIOException("Failed to write JSON to " + file, e);
    }
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  /**
   * Formats a parameter map as a human-readable string (e.g., {@code "M=32, efSearch=128"}). Keys
   * are emitted in lexicographic order so that the resulting string is stable across runs (the
   * checkpointing store uses this as part of its canonical key).
   */
  public static String formatParams(Map<String, String> params) {
    if (params.isEmpty()) return "default";
    return params.entrySet().stream()
        .sorted(Map.Entry.comparingByKey())
        .map(e -> e.getKey() + "=" + e.getValue())
        .collect(Collectors.joining(", "));
  }

  private static String csvEscape(String s) {
    if (s.indexOf(',') >= 0 || s.indexOf('"') >= 0 || s.indexOf('\n') >= 0) {
      return "\"" + s.replace("\"", "\"\"") + "\"";
    }
    return s;
  }

  private static String jsonEscape(String s) {
    return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
  }

  private static String mapToJson(Map<String, String> map) {
    if (map.isEmpty()) return "{}";
    StringBuilder sb = new StringBuilder("{ ");
    int i = 0;
    for (var e : map.entrySet()) {
      if (i++ > 0) sb.append(", ");
      sb.append('"').append(jsonEscape(e.getKey())).append("\": \"");
      sb.append(jsonEscape(e.getValue())).append('"');
    }
    sb.append(" }");
    return sb.toString();
  }
}
