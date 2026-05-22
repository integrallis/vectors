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
package com.integrallis.vectors.optimizer.study;

import com.integrallis.vectors.bench.report.LatencyCollector;
import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.VectorCollectionBuilder;
import com.integrallis.vectors.optimizer.data.Qrels;
import com.integrallis.vectors.optimizer.data.Queries;
import com.integrallis.vectors.optimizer.data.Run;
import com.integrallis.vectors.optimizer.embed.EmbeddingProvider;
import com.integrallis.vectors.optimizer.eval.Metrics;
import com.integrallis.vectors.optimizer.objective.Objective;
import com.integrallis.vectors.optimizer.space.Trial;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Builds a {@link VectorCollection} from a single {@link Trial}, ingests the configured corpus,
 * runs warmup + measurement search rounds, and produces a {@link TrialResult} with IR metrics,
 * latency percentiles, build cost, memory footprint, and a composite objective score.
 *
 * <p>One trial owns one collection — the {@code try-with-resources} ensures the collection is
 * closed even when the run throws. Sequential rounds avoid contending with other trials on memory
 * or JVM warmup.
 */
public final class IndexStudy {

  private final StudyConfig cfg;
  private final EmbeddingProvider embedder;

  public IndexStudy(StudyConfig cfg, EmbeddingProvider embedder) {
    this.cfg = Objects.requireNonNull(cfg, "cfg");
    this.embedder = Objects.requireNonNull(embedder, "embedder");
  }

  /** Executes one trial. The collection is built, scored, and closed before the result returns. */
  public TrialResult runOne(Trial trial) {
    Objects.requireNonNull(trial, "trial");
    Instant startedAt = Instant.now();
    List<Document> corpus = cfg.corpusSource().get();
    if (corpus.isEmpty()) throw new IllegalStateException("corpusSource produced empty corpus");
    int dim = corpus.get(0).vector().length;

    Queries queries = cfg.queriesSource().get();
    Qrels qrels = cfg.qrelsSource().get();
    int k = cfg.kForMetrics();

    VectorCollectionBuilder cb = configureBuilder(trial, dim);
    long buildStart = System.nanoTime();
    long buildTimeMs;
    long memoryBytes;
    LatencyCollector lc =
        new LatencyCollector(Math.max(1, queries.size() * cfg.measurementRounds()));
    Run run;

    try (VectorCollection col = cb.build()) {
      col.addAll(corpus);
      col.commit();
      buildTimeMs = (System.nanoTime() - buildStart) / 1_000_000L;
      memoryBytes = (long) col.physicalSize() * (long) col.config().dimension() * 4L;

      List<String> qIds = new ArrayList<>(queries.byId().keySet());
      List<String> qTexts = new ArrayList<>(queries.byId().values());
      List<float[]> qVecs = embedder.embedAll(qTexts);

      int efSearch = optInt(trial.params(), "efSearch", -1);

      // Warmup rounds: discarded.
      for (int r = 0; r < cfg.warmupRounds(); r++) {
        for (float[] q : qVecs) {
          col.search(buildSearchRequest(q, k, efSearch));
        }
      }

      // Measurement rounds: latency from every round, ranking from the last round.
      Run.Builder rb = Run.builder();
      for (int r = 0; r < cfg.measurementRounds(); r++) {
        boolean lastRound = r == cfg.measurementRounds() - 1;
        for (int i = 0; i < qVecs.size(); i++) {
          long t0 = System.nanoTime();
          SearchResult sr = col.search(buildSearchRequest(qVecs.get(i), k, efSearch));
          lc.record(System.nanoTime() - t0);
          if (lastRound) {
            for (SearchResult.Hit h : sr.hits()) rb.add(qIds.get(i), h.id(), h.score());
          }
        }
      }
      run = rb.build();
    }
    if (cfg.gcBetweenTrials()) System.gc();

    lc.compute();
    double recall = Metrics.recallAtK(qrels, run, k);
    double ndcg = Metrics.ndcgAtK(qrels, run, k);
    double precision = Metrics.precisionAtK(qrels, run, k);
    double f1 = Metrics.f1AtK(qrels, run, k);
    double mrr = Metrics.mrr(qrels, run);
    double obj =
        Objective.score(
            recall,
            ndcg,
            precision,
            f1,
            mrr,
            lc.p95Us(),
            buildTimeMs,
            memoryBytes,
            cfg.objectiveWeights());
    return new TrialResult(
        trial,
        startedAt,
        Instant.now(),
        recall,
        ndcg,
        precision,
        f1,
        mrr,
        lc.p50Us(),
        lc.p95Us(),
        lc.p99Us(),
        buildTimeMs,
        memoryBytes,
        obj);
  }

  private static SearchRequest buildSearchRequest(float[] query, int k, int efSearch) {
    SearchRequest.Builder b = SearchRequest.builder(query, k);
    if (efSearch > 0) b.searchListSize(efSearch);
    return b.includeVector(false).includeText(false).includeMetadata(false).build();
  }

  // --- Trial → Builder mapping ---

  private VectorCollectionBuilder configureBuilder(Trial trial, int dim) {
    Map<String, Object> p = trial.params();
    SimilarityFunction metric =
        enumOf(p, "metric", SimilarityFunction.class, SimilarityFunction.COSINE);
    IndexType indexType = enumOf(p, "indexType", IndexType.class, IndexType.HNSW);
    QuantizerKind quantizer = enumOf(p, "quantizer", QuantizerKind.class, QuantizerKind.NONE);

    VectorCollectionBuilder b =
        VectorCollection.builder()
            .dimension(dim)
            .metric(metric)
            .indexType(indexType)
            .quantizer(quantizer);
    if (indexType == IndexType.HNSW) {
      int m = optInt(p, "m", VectorCollectionBuilder.DEFAULT_HNSW_M);
      int ef = optInt(p, "efConstruction", VectorCollectionBuilder.DEFAULT_HNSW_EF_CONSTRUCTION);
      b.hnswM(m).hnswEfConstruction(ef);
    } else if (indexType == IndexType.VAMANA) {
      int r = optInt(p, "vamanaR", VectorCollectionBuilder.DEFAULT_VAMANA_R);
      int l = optInt(p, "vamanaL", VectorCollectionBuilder.DEFAULT_VAMANA_L);
      double alpha = optDouble(p, "vamanaAlpha", VectorCollectionBuilder.DEFAULT_VAMANA_ALPHA);
      b.vamanaMaxDegree(r).vamanaSearchListSize(l).vamanaAlpha((float) alpha);
    }
    return b;
  }

  private static int optInt(Map<String, Object> p, String key, int defaultValue) {
    Object v = p.get(key);
    if (v == null) return defaultValue;
    return ((Number) v).intValue();
  }

  private static double optDouble(Map<String, Object> p, String key, double defaultValue) {
    Object v = p.get(key);
    if (v == null) return defaultValue;
    return ((Number) v).doubleValue();
  }

  private static <E extends Enum<E>> E enumOf(
      Map<String, Object> p, String key, Class<E> type, E defaultValue) {
    Object v = p.get(key);
    if (v == null) return defaultValue;
    if (type.isInstance(v)) return type.cast(v);
    return Enum.valueOf(type, v.toString().toUpperCase(Locale.ROOT));
  }
}
