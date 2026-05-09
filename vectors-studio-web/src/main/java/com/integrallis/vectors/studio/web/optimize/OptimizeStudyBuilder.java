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
package com.integrallis.vectors.studio.web.optimize;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.optimizer.data.MetadataQrelsDeriver;
import com.integrallis.vectors.optimizer.data.Qrels;
import com.integrallis.vectors.optimizer.data.Queries;
import com.integrallis.vectors.optimizer.embed.EmbeddingProvider;
import com.integrallis.vectors.optimizer.objective.ObjectiveWeights;
import com.integrallis.vectors.optimizer.space.ParamSpec;
import com.integrallis.vectors.optimizer.space.SearchSpace;
import com.integrallis.vectors.optimizer.study.StudyConfig;
import com.integrallis.vectors.studio.core.StudioSession;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Translates an {@link OptimizeRequestDto} plus a Studio collection into a fully-formed {@link
 * StudyConfig} + matching {@link EmbeddingProvider}. Strategy:
 *
 * <ol>
 *   <li>Pull every {@code (id, vector)} pair from the backend via {@code streamAllVectors}.
 *   <li>Sample {@code querySampleSize} docs as evaluation queries; their existing vectors become
 *       the embeddings (the embedder is a fixed {@code text -> vector} lookup).
 *   <li>If a metadata field is supplied, derive qrels via {@link MetadataQrelsDeriver}; otherwise
 *       compute a "self-similarity" gold standard by FLAT-searching the original corpus for each
 *       query's top-k neighbours.
 *   <li>Pin the search space to HNSW with caller-controlled bounds on {@code m} and
 *       {@code efConstruction}.
 * </ol>
 */
public final class OptimizeStudyBuilder {

  /** Bundles the StudyConfig with the embedder it expects to be paired with. */
  public record Built(StudyConfig config, EmbeddingProvider embedder) {}

  private OptimizeStudyBuilder() {}

  public static Built build(StudioSession session, OptimizeRequestDto dto) {
    List<Document> corpus = collectCorpus(session, dto.collection());
    if (corpus.isEmpty()) {
      throw new IllegalArgumentException("collection '" + dto.collection() + "' is empty");
    }

    Random rng = new Random(dto.seed());
    List<Document> queryDocs = new ArrayList<>(corpus);
    Collections.shuffle(queryDocs, rng);
    int nQueries = Math.min(Math.max(1, dto.querySampleSize()), queryDocs.size());
    queryDocs = queryDocs.subList(0, nQueries);

    Map<String, String> queries = new LinkedHashMap<>();
    Map<String, float[]> embeddings = new LinkedHashMap<>();
    for (int i = 0; i < queryDocs.size(); i++) {
      Document d = queryDocs.get(i);
      String qid = "q-" + i;
      queries.put(qid, d.id());
      embeddings.put(d.id(), d.vector());
    }
    EmbeddingProvider embedder = embeddings::get;

    Qrels qrels =
        dto.metadataField() != null && !dto.metadataField().isBlank()
            ? metadataQrels(corpus, queries, dto.metadataField())
            : selfSimilarityQrels(corpus, queryDocs, queries, dto.kForMetrics());

    SearchSpace space =
        new SearchSpace(
            List.of(
                new ParamSpec.FixedString("metric", "COSINE"),
                new ParamSpec.FixedString("indexType", "HNSW"),
                new ParamSpec.IntRange("m", dto.mMin(), dto.mMax(), true),
                new ParamSpec.IntRange("efConstruction", dto.efMin(), dto.efMax(), true)));

    ObjectiveWeights weights =
        ObjectiveWeights.builder()
            .kForMetrics(dto.kForMetrics())
            .recallWeight(dto.recallWeight())
            .ndcgWeight(dto.ndcgWeight())
            .latencyP95Weight(dto.latencyWeight())
            .build();

    StudyConfig.SamplerKind sk = StudyConfig.SamplerKind.valueOf(dto.sampler().toUpperCase());
    final List<Document> finalCorpus = corpus;
    final Qrels finalQrels = qrels;
    final Queries finalQueries = new Queries(queries);
    StudyConfig cfg =
        StudyConfig.builder()
            .searchSpace(space)
            .objectiveWeights(weights)
            .samplerKind(sk)
            .corpusSource(() -> finalCorpus)
            .qrelsSource(() -> finalQrels)
            .queriesSource(() -> finalQueries)
            .nTrials(dto.nTrials())
            .kForMetrics(dto.kForMetrics())
            .seed(dto.seed())
            .build();
    return new Built(cfg, embedder);
  }

  private static List<Document> collectCorpus(StudioSession session, String collectionName) {
    List<Document> corpus = new ArrayList<>();
    session
        .backend()
        .streamAllVectors(
            collectionName, (id, v) -> corpus.add(Document.of(id, v.clone())), null);
    return corpus;
  }

  private static Qrels metadataQrels(
      List<Document> corpus, Map<String, String> queries, String field) {
    var derived = MetadataQrelsDeriver.derive(corpus, field).qrels();
    Map<String, Map<String, Integer>> filtered = new LinkedHashMap<>();
    for (var e : queries.entrySet()) {
      String docId = e.getValue();
      var row = derived.relevance().get(docId);
      if (row != null) filtered.put(e.getKey(), row);
    }
    return new Qrels(filtered);
  }

  private static Qrels selfSimilarityQrels(
      List<Document> corpus, List<Document> queryDocs, Map<String, String> queries, int k) {
    Map<String, Map<String, Integer>> rel = new LinkedHashMap<>();
    int dim = corpus.get(0).vector().length;
    try (VectorCollection ref =
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .build()) {
      ref.addAll(corpus);
      ref.commit();
      List<String> qIds = new ArrayList<>(queries.keySet());
      for (int i = 0; i < queryDocs.size(); i++) {
        SearchResult sr =
            ref.search(SearchRequest.builder(queryDocs.get(i).vector(), k + 1).build());
        Map<String, Integer> row = new LinkedHashMap<>();
        Set<String> seen = new java.util.HashSet<>();
        for (SearchResult.Hit h : sr.hits()) {
          if (h.id().equals(queryDocs.get(i).id())) continue;
          if (seen.add(h.id())) row.put(h.id(), 1);
          if (row.size() >= k) break;
        }
        if (!row.isEmpty()) rel.put(qIds.get(i), row);
      }
    }
    return new Qrels(rel);
  }
}
