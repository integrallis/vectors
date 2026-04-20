package com.integrallis.vectors.bench;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.integrallis.vectors.bench.report.BenchmarkReporter;
import com.integrallis.vectors.bench.report.BenchmarkResult;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class CheckpointStoreTest {

  private static Path csvOf(Path dir) {
    return dir.resolve("results.csv");
  }

  private static Path markersOf(Path dir) {
    return dir.resolve("markers");
  }

  private static BenchmarkResult row(
      String algo, Map<String, String> build, Map<String, String> search) {
    return BenchmarkResult.builder("sift-128-euclidean", algo)
        .buildParams(build)
        .searchParams(search)
        .recall10(0.95)
        .qps(1000)
        .p50Us(100)
        .p95Us(200)
        .p99Us(300)
        .buildTimeSeconds(1.0)
        .indexSizeMb(10.0)
        .build();
  }

  @Test
  void freshStoreReportsNothingCompleted(@TempDir Path tmp) {
    CheckpointStore store = new CheckpointStore(csvOf(tmp), markersOf(tmp));
    assertFalse(store.isRowCompleted("sift-128-euclidean", "hnsw", Map.of("M", "16"), Map.of()));
    assertFalse(store.isBuildCompleted("sift-128-euclidean", "hnsw", Map.of("M", "16")));
  }

  @Test
  void appendedRowIsDetectedByFreshStore(@TempDir Path tmp) {
    Path csv = csvOf(tmp);
    Map<String, String> build = Map.of("M", "16", "efConstruction", "100");
    Map<String, String> search = Map.of("efSearch", "64");
    BenchmarkReporter.csvAppend(row("hnsw", build, search), csv);

    CheckpointStore store = new CheckpointStore(csv, markersOf(tmp));
    assertTrue(store.isRowCompleted("sift-128-euclidean", "hnsw", build, search));
    assertFalse(
        store.isRowCompleted("sift-128-euclidean", "hnsw", build, Map.of("efSearch", "128")));
    assertFalse(store.isBuildCompleted("sift-128-euclidean", "hnsw", build));
  }

  @Test
  void markerFileSignalsCompletedBuildSweep(@TempDir Path tmp) throws IOException {
    Path csv = csvOf(tmp);
    Map<String, String> build = Map.of("M", "16", "efConstruction", "100");

    CheckpointStore store = new CheckpointStore(csv, markersOf(tmp));
    store.markBuildCompleted("sift-128-euclidean", "hnsw", build);
    assertTrue(Files.isDirectory(markersOf(tmp)));
    try (var stream = Files.list(markersOf(tmp))) {
      assertTrue(
          stream.anyMatch(p -> p.getFileName().toString().endsWith(".completed")),
          "expected a .completed marker in " + markersOf(tmp));
    }

    CheckpointStore reloaded = new CheckpointStore(csv, markersOf(tmp));
    assertTrue(reloaded.isBuildCompleted("sift-128-euclidean", "hnsw", build));
  }

  @Test
  void rowsWithDifferentSearchParamsAreTrackedIndependently(@TempDir Path tmp) {
    Path csv = csvOf(tmp);
    Map<String, String> build = Map.of("R", "32", "L_build", "100", "alpha", "1.2");
    BenchmarkReporter.csvAppend(row("vamana", build, Map.of("L_search", "64")), csv);
    BenchmarkReporter.csvAppend(row("vamana", build, Map.of("L_search", "128")), csv);

    CheckpointStore store = new CheckpointStore(csv, markersOf(tmp));
    assertTrue(
        store.isRowCompleted("sift-128-euclidean", "vamana", build, Map.of("L_search", "64")));
    assertTrue(
        store.isRowCompleted("sift-128-euclidean", "vamana", build, Map.of("L_search", "128")));
    assertFalse(
        store.isRowCompleted("sift-128-euclidean", "vamana", build, Map.of("L_search", "256")));
  }

  @Test
  void csvParserHandlesQuotedFieldsContainingCommas(@TempDir Path tmp) {
    Path csv = csvOf(tmp);
    // build_params with a comma-separated key=value will be quoted by csvEscape.
    Map<String, String> build = Map.of("M", "16", "efConstruction", "100");
    Map<String, String> search = Map.of("efSearch", "64");
    BenchmarkReporter.csvAppend(row("hnsw", build, search), csv);

    List<String> lines;
    try {
      lines = Files.readAllLines(csv);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    // Header + 1 data row.
    assertTrue(lines.size() >= 2);
    assertTrue(lines.get(1).contains("\"")); // quoted because of comma in formatted params

    CheckpointStore store = new CheckpointStore(csv, markersOf(tmp));
    assertTrue(store.isRowCompleted("sift-128-euclidean", "hnsw", build, search));
  }

  @Test
  void markRowCompletedIsReflectedInIsRowCompleted(@TempDir Path tmp) {
    CheckpointStore store = new CheckpointStore(csvOf(tmp), markersOf(tmp));
    Map<String, String> build = Map.of("M", "16", "efConstruction", "100");
    Map<String, String> search = Map.of("efSearch", "64");
    assertFalse(store.isRowCompleted("sift-128-euclidean", "hnsw", build, search));
    store.markRowCompleted("sift-128-euclidean", "hnsw", build, search);
    assertTrue(store.isRowCompleted("sift-128-euclidean", "hnsw", build, search));
  }
}
