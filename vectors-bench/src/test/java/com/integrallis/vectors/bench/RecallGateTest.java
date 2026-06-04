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
package com.integrallis.vectors.bench;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.fail;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.VectorCollectionBuilder;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Consumer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * P1.1 — recall@10 regression GATE. Deterministic, downloads nothing, and runs against the
 * PERSISTENT (committed-then-reopened) collection path so the gate exercises the same mapped
 * adapters users serve from — the path where the P1.3a IVF over-query regression actually lived.
 *
 * <p>For each gated (index, params) row it builds a collection on a fixed synthetic corpus, commits
 * it to disk, reopens it, runs the query set, and compares recall@10 against the committed baseline
 * in {@code recall-baseline.txt}. A drop greater than the per-row tolerance (default {@value
 * #DEFAULT_TOLERANCE}) fails the build; an improvement never fails. The baseline is recorded on CI
 * (or a 256-bit-capped x86 host) and committed.
 *
 * <p>Modes (system property {@code recall.gate.mode}):
 *
 * <ul>
 *   <li>{@code verify} (default) — diff measured recall against the committed baseline; fail on a
 *       drop beyond tolerance or a missing baseline row.
 *   <li>{@code record} — print a fresh baseline file to stdout (assert nothing); copy it into
 *       {@code src/test/resources/recall-baseline.txt} after an intentional algorithm change.
 * </ul>
 *
 * <p>Determinism: every graph/cluster build pins its seed and uses a single build thread (HNSW and
 * Vamana are only bit-exact single-threaded). The P1.5 256-bit SIMD cap keeps recall consistent
 * across x86 AVX2/AVX-512 runners. Tagged {@code recall} so it runs only via the dedicated {@code
 * recallGate} task, not the default {@code test}/{@code build}.
 */
@Tag("recall")
class RecallGateTest {

  private static final int N = 10_000;
  private static final int DIM = 128;
  private static final int N_QUERIES = 200;
  private static final int K = 10;
  private static final long SEED = 42L;
  private static final float DEFAULT_TOLERANCE = 0.005f;
  private static final String BASELINE_RESOURCE = "/recall-baseline.txt";

  /** A single gated measurement: an index build + a (label, beam-width, over-query) search row. */
  private record Row(String index, String params, int searchListSize, float overQueryFactor) {
    String key() {
      return index + " " + params;
    }
  }

  /** An index build recipe plus the search rows measured against it. */
  private record GateIndex(
      String name, IndexType type, Consumer<VectorCollectionBuilder> config, List<Row> rows) {}

  private static final List<GateIndex> GATED =
      List.of(
          new GateIndex(
              "HNSW",
              IndexType.HNSW,
              b -> b.hnswM(16).hnswEfConstruction(200).hnswBuildThreads(1),
              List.of(
                  new Row("HNSW", "efSearch=128", 128, 1.0f),
                  new Row("HNSW", "efSearch=200", 200, 1.0f))),
          new GateIndex(
              "VAMANA",
              IndexType.VAMANA,
              b ->
                  b.vamanaMaxDegree(64)
                      .vamanaSearchListSize(128)
                      .vamanaAlpha(1.2f)
                      .vamanaSeed(SEED)
                      .vamanaBuildThreads(1),
              List.of(
                  new Row("VAMANA", "L=128", 128, 1.0f), new Row("VAMANA", "L=200", 200, 1.0f))),
          new GateIndex(
              "IVF_FLAT",
              IndexType.IVF_FLAT,
              b -> b.ivfK(64).ivfNprobe(8).ivfSeed(SEED),
              List.of(new Row("IVF_FLAT", "nprobe=8,oq=1.0", 8, 1.0f))),
          new GateIndex(
              "IVF_PQ",
              IndexType.IVF_PQ,
              b -> b.ivfK(64).ivfNprobe(8).ivfPqSubspaces(16).ivfRescoreFactor(4).ivfSeed(SEED),
              List.of(
                  new Row("IVF_PQ", "nprobe=8,m=16,rescore=4,oq=1.0", 8, 1.0f),
                  new Row("IVF_PQ", "nprobe=8,m=16,rescore=4,oq=4.0", 8, 4.0f))));

  @TempDir static Path tmp;

  private static float[][] corpus;
  private static float[][] queries;
  private static int[][] groundTruth;
  private static final Map<String, VectorCollection> REOPENED = new LinkedHashMap<>();

  @BeforeAll
  static void buildCorpusAndIndexes() {
    Random rng = new Random(SEED);
    corpus = new float[N][DIM];
    for (float[] v : corpus) {
      for (int d = 0; d < DIM; d++) {
        v[d] = rng.nextFloat() * 2f - 1f;
      }
    }
    queries = new float[N_QUERIES][DIM];
    for (float[] q : queries) {
      for (int d = 0; d < DIM; d++) {
        q[d] = rng.nextFloat() * 2f - 1f;
      }
    }
    groundTruth =
        RecallUtil.bruteForceGroundTruth(queries, corpus, SimilarityFunction.EUCLIDEAN, K);

    for (GateIndex gi : GATED) {
      Path dir = tmp.resolve(gi.name());
      // Build, commit, and close so the in-memory adapter is released ...
      try (VectorCollection col = open(dir, gi)) {
        for (int i = 0; i < N; i++) {
          col.add(Document.of("doc-" + i, corpus[i]));
        }
        col.commit();
      }
      // ... then reopen from disk so search runs through the persistent mapped adapter.
      REOPENED.put(gi.name(), open(dir, gi));
    }
  }

  @AfterAll
  static void closeIndexes() {
    for (VectorCollection col : REOPENED.values()) {
      col.close();
    }
    REOPENED.clear();
  }

  private static VectorCollection open(Path dir, GateIndex gi) {
    VectorCollectionBuilder b =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(gi.type())
            .storagePath(dir);
    gi.config().accept(b);
    return b.build();
  }

  private static double measureRecall(GateIndex gi, Row row) {
    VectorCollection col = REOPENED.get(gi.name());
    int[][] approx = new int[queries.length][];
    for (int i = 0; i < queries.length; i++) {
      List<SearchResult.Hit> hits =
          col.search(
                  SearchRequest.builder(queries[i], K)
                      .searchListSize(row.searchListSize())
                      .overQueryFactor(row.overQueryFactor())
                      .build())
              .hits();
      int[] ords = new int[hits.size()];
      for (int j = 0; j < hits.size(); j++) {
        ords[j] = Integer.parseInt(hits.get(j).id().substring("doc-".length()));
      }
      approx[i] = ords;
    }
    return RecallUtil.meanRecallAtK(groundTruth, approx, K);
  }

  @Test
  void recallMatchesBaselineOrRecord() {
    Map<String, Double> measured = new LinkedHashMap<>();
    for (GateIndex gi : GATED) {
      for (Row row : gi.rows()) {
        measured.put(row.key(), measureRecall(gi, row));
      }
    }

    if ("record".equalsIgnoreCase(System.getProperty("recall.gate.mode"))) {
      System.out.println(renderBaseline(measured));
      return; // record mode asserts nothing
    }

    Map<String, float[]> baseline = loadBaseline(); // key -> [recall, tolerance]
    List<String> failures = new ArrayList<>();
    for (Map.Entry<String, Double> e : measured.entrySet()) {
      String key = e.getKey();
      double now = e.getValue();
      float[] base = baseline.get(key);
      if (base == null) {
        failures.add(
            "no baseline row for \""
                + key
                + "\" (measured "
                + fmt(now)
                + ") — run -Drecall.gate.mode=record and commit recall-baseline.txt");
        continue;
      }
      double expected = base[0];
      double tol = base[1];
      double drop = expected - now;
      if (drop > tol) {
        failures.add(
            String.format(
                "%-34s baseline=%.4f measured=%.4f drop=%.4f > tol=%.4f",
                key, expected, now, drop, tol));
      }
    }

    if (!failures.isEmpty()) {
      fail(
          "recall@10 regression vs baseline (run -Drecall.gate.mode=record to re-baseline an"
              + " intentional change):\n  "
              + String.join("\n  ", failures));
    }
    // Sanity: every committed baseline row must have been measured (catches a stale baseline row
    // for a gated index that was removed or renamed).
    for (String key : baseline.keySet()) {
      assertThat(measured)
          .as("baseline row \"%s\" has no corresponding measurement", key)
          .containsKey(key);
    }
  }

  private static String renderBaseline(Map<String, Double> measured) {
    StringBuilder sb = new StringBuilder();
    sb.append("# recall-baseline.txt — P1.1 recall@10 regression gate\n");
    sb.append(
        "# Recorded by RecallGateTest in record mode. Persistent (committed+reopened) path.\n");
    sb.append(
        "# corpus: n="
            + N
            + " dim="
            + DIM
            + " queries="
            + N_QUERIES
            + " k="
            + K
            + " seed="
            + SEED
            + " metric=EUCLIDEAN\n");
    sb.append("# tolerance: " + DEFAULT_TOLERANCE + " (global; optional 4th column overrides)\n");
    sb.append("# format: <index> <params> <recall@10> [<tolerance>]\n");
    for (Map.Entry<String, Double> e : measured.entrySet()) {
      sb.append(String.format("%s %.4f%n", e.getKey(), e.getValue()));
    }
    return sb.toString();
  }

  private static Map<String, float[]> loadBaseline() {
    Map<String, float[]> out = new LinkedHashMap<>();
    try (InputStream in = RecallGateTest.class.getResourceAsStream(BASELINE_RESOURCE)) {
      if (in == null) {
        throw new IllegalStateException(
            "missing baseline resource " + BASELINE_RESOURCE + " — run -Drecall.gate.mode=record");
      }
      String text = new String(in.readAllBytes(), StandardCharsets.UTF_8);
      for (String raw : text.split("\\R")) {
        String line = raw.strip();
        if (line.isEmpty() || line.startsWith("#")) {
          continue;
        }
        String[] t = line.split("\\s+");
        if (t.length < 3) {
          throw new IllegalStateException("malformed baseline line: " + raw);
        }
        // key is "<index> <params>"; recall is the 3rd token; optional 4th is per-row tolerance.
        String key = t[0] + " " + t[1];
        float recall = Float.parseFloat(t[2]);
        float tol = t.length >= 4 ? Float.parseFloat(t[3]) : DEFAULT_TOLERANCE;
        out.put(key, new float[] {recall, tol});
      }
    } catch (java.io.IOException ex) {
      throw new IllegalStateException("failed to read " + BASELINE_RESOURCE, ex);
    }
    return out;
  }

  private static String fmt(double v) {
    return String.format("%.4f", v);
  }
}
