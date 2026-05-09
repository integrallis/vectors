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
package com.integrallis.vectors.demo.optimizer;

import com.integrallis.vectors.bench.dataset.DatasetDownloader;
import com.integrallis.vectors.bench.dataset.Hdf5Loader;
import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.optimizer.data.Qrels;
import com.integrallis.vectors.optimizer.data.Queries;
import com.integrallis.vectors.optimizer.embed.EmbeddingProvider;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Loads an ANN-Benchmarks HDF5 dataset (auto-downloaded on first run) and exposes it in the shape
 * the optimizer expects: a {@code List<Document>} corpus, a {@link Queries} map, ground-truth
 * {@link Qrels}, and an {@link EmbeddingProvider} that resolves query IDs back to their precomputed
 * vectors.
 *
 * <p>The default dataset is {@code fashion-mnist-784-euclidean} (~30 MB). Override via the {@code
 * VECTORS_OPTIMIZER_TUTORIAL_DATASET} environment variable — for example {@code sift-128-euclidean}
 * or {@code glove-100-angular}.
 *
 * <p>Subsamples to {@code 10_000} corpus vectors and {@code 200} queries by default for fast
 * iteration. Set {@code VECTORS_OPTIMIZER_TUTORIAL_FULL=1} (or pass {@code --full} to a stage
 * runner) to load the full dataset.
 */
public final class TutorialDataset {

  /** Default dataset name used when no override is supplied. */
  public static final String DEFAULT_DATASET = "fashion-mnist-784-euclidean";

  /**
   * Default corpus subsample size. Sized so each HNSW build completes in ~5 s, keeping a 20-trial
   * sweep under two minutes. With a 10K corpus HNSW reaches 100% recall at every parameter setting
   * — the optimization story is therefore minimising build time and search latency at constant
   * recall, which is the exact problem most production teams actually face.
   */
  public static final int DEFAULT_CORPUS_LIMIT = 10_000;

  /** Default query subsample size. */
  public static final int DEFAULT_QUERY_LIMIT = 200;

  private final String datasetName;
  private final List<Document> corpus;
  private final Queries queries;
  private final Qrels qrels;
  private final Map<String, float[]> queryVectorsById;
  private final int dimension;

  private TutorialDataset(
      String datasetName,
      List<Document> corpus,
      Queries queries,
      Qrels qrels,
      Map<String, float[]> queryVectorsById,
      int dimension) {
    this.datasetName = datasetName;
    this.corpus = corpus;
    this.queries = queries;
    this.qrels = qrels;
    this.queryVectorsById = queryVectorsById;
    this.dimension = dimension;
  }

  /** Loads the dataset honouring the {@code VECTORS_OPTIMIZER_TUTORIAL_*} environment overrides. */
  public static TutorialDataset loadDefault() {
    String name =
        System.getenv().getOrDefault("VECTORS_OPTIMIZER_TUTORIAL_DATASET", DEFAULT_DATASET);
    boolean full = "1".equals(System.getenv("VECTORS_OPTIMIZER_TUTORIAL_FULL"));
    return load(
        name,
        full ? Integer.MAX_VALUE : DEFAULT_CORPUS_LIMIT,
        full ? Integer.MAX_VALUE : DEFAULT_QUERY_LIMIT);
  }

  /**
   * Loads the dataset, capping corpus and query counts as requested. {@code Integer.MAX_VALUE} =
   * full.
   */
  public static TutorialDataset load(String datasetName, int corpusLimit, int queryLimit) {
    Path hdf5;
    try {
      hdf5 = DatasetDownloader.ensureAvailable(datasetName);
    } catch (IOException ioe) {
      throw new UncheckedIOException("Failed to materialise dataset " + datasetName, ioe);
    }

    float[][] train = Hdf5Loader.readTrainVectors(hdf5);
    float[][] test = Hdf5Loader.readTestVectors(hdf5);
    // /neighbors is the original GT; intentionally unused — we recompute against the subsample.

    int nCorpus = Math.min(train.length, corpusLimit);
    int nQuery = Math.min(test.length, queryLimit);
    int dim = train[0].length;

    // Stride sampling avoids class bias when the source dataset is sorted by label.
    int stride = Math.max(1, train.length / nCorpus);
    List<Document> docs = new ArrayList<>(nCorpus);
    for (int i = 0; i < nCorpus; i++) {
      int origIdx = Math.min(train.length - 1, i * stride);
      docs.add(Document.of("doc-" + origIdx, train[origIdx]));
    }

    Map<String, String> qById = new LinkedHashMap<>();
    Map<String, float[]> qVecs = new LinkedHashMap<>();
    for (int q = 0; q < nQuery; q++) {
      String qid = "q-" + q;
      qById.put(qid, qid);
      qVecs.put(qid, test[q]);
    }

    Map<String, Map<String, Integer>> rel = computeFlatGroundTruth(docs, qById, qVecs, dim, 10);
    return new TutorialDataset(datasetName, docs, new Queries(qById), new Qrels(rel), qVecs, dim);
  }

  /**
   * Recomputes ground truth on the subsample with a brute-force FLAT search. This is necessary
   * because the dataset's bundled {@code /neighbors} are computed against the FULL corpus; once we
   * subsample, those ordinals may be missing or no longer represent the actual top-k inside our
   * smaller corpus, which artificially caps recall@k to the survival rate.
   */
  private static Map<String, Map<String, Integer>> computeFlatGroundTruth(
      List<Document> docs,
      Map<String, String> queryIds,
      Map<String, float[]> queryVectors,
      int dim,
      int k) {
    Map<String, Map<String, Integer>> out = new LinkedHashMap<>();
    try (VectorCollection flat =
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.FLAT)
            .build()) {
      flat.addAll(docs);
      flat.commit();
      for (String qid : queryIds.keySet()) {
        SearchResult sr =
            flat.search(
                SearchRequest.builder(queryVectors.get(qid), k)
                    .includeVector(false)
                    .includeText(false)
                    .includeMetadata(false)
                    .build());
        Map<String, Integer> grades = new LinkedHashMap<>();
        int rank = 0;
        for (SearchResult.Hit h : sr.hits()) {
          grades.put(h.id(), Math.max(1, k - rank));
          rank++;
        }
        out.put(qid, grades);
      }
    }
    return out;
  }

  /** Embedding provider that maps query IDs (used as text) back to their precomputed vectors. */
  public EmbeddingProvider embedder() {
    return text -> {
      float[] v = queryVectorsById.get(text);
      if (v == null) throw new IllegalArgumentException("Unknown query id: " + text);
      return v;
    };
  }

  public String datasetName() {
    return datasetName;
  }

  public List<Document> corpus() {
    return corpus;
  }

  public Queries queries() {
    return queries;
  }

  public Qrels qrels() {
    return qrels;
  }

  public int dimension() {
    return dimension;
  }
}
