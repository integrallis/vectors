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
package com.integrallis.vectors.studio.web.routing;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.hybrid.MaximalMarginalRelevance;
import com.integrallis.vectors.optimizer.embed.EmbeddingProvider;
import com.integrallis.vectors.studio.core.StudioSession;
import com.integrallis.vectors.studio.core.projection.ProjectionAlgorithm;
import com.integrallis.vectors.studio.core.projection.ProjectionParams;
import com.integrallis.vectors.studio.core.projection.ProjectionRequest;
import com.integrallis.vectors.studio.core.search.SearchHit;
import com.integrallis.vectors.studio.web.dataset.DatasetCatalog;
import com.integrallis.vectors.studio.web.dataset.DatasetCatalogEntry;
import com.integrallis.vectors.studio.web.dto.ProjectionRequestDto;
import com.integrallis.vectors.studio.web.embed.ProviderRegistry;
import com.integrallis.vectors.studio.web.projection.ProjectionEvent;
import com.integrallis.vectors.studio.web.projection.ProjectionJob;
import com.integrallis.vectors.studio.web.projection.ProjectionJobManager;
import io.helidon.http.HeaderNames;
import io.helidon.http.Status;
import io.helidon.http.sse.SseEvent;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import io.helidon.webserver.sse.SseSink;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Flow;

/** JSON + SSE API consumed by the projector island. */
public final class ApiRoutes implements HttpService {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  private final StudioSession session;
  private final ProjectionJobManager jobs;
  private final DatasetCatalog catalog;
  private final ProviderRegistry providerRegistry;

  public ApiRoutes(
      StudioSession session,
      ProjectionJobManager jobs,
      DatasetCatalog catalog,
      ProviderRegistry providerRegistry) {
    this.session = session;
    this.jobs = jobs;
    this.catalog = catalog;
    this.providerRegistry = providerRegistry;
  }

  @Override
  public void routing(HttpRules rules) {
    rules
        .post("/api/projections", this::submit)
        .get("/api/projections/{id}/events", this::events)
        .delete("/api/projections/{id}", this::cancel)
        .get("/api/collections/{name}/tensors.bytes", this::tensors)
        .get("/api/collections/{name}/metadata.tsv", this::metadata)
        .post("/api/collections/{name}/search", this::knnSearch);
  }

  private void submit(ServerRequest req, ServerResponse res) {
    try {
      ProjectionRequestDto dto =
          MAPPER.readValue(req.content().inputStream(), ProjectionRequestDto.class);
      List<String> ids = new ArrayList<>();
      List<float[]> rows = new ArrayList<>();
      session
          .backend()
          .streamAllVectors(
              dto.collection(),
              (id, v) -> {
                ids.add(id);
                rows.add(v);
              },
              null);
      if (dto.sampleSize() > 0 && dto.sampleSize() < rows.size()) {
        java.util.Random rng = new java.util.Random(0xC0FFEE);
        for (int i = rows.size() - 1; i > 0; i--) {
          int j = rng.nextInt(i + 1);
          var tmpId = ids.get(i);
          ids.set(i, ids.get(j));
          ids.set(j, tmpId);
          var tmpV = rows.get(i);
          rows.set(i, rows.get(j));
          rows.set(j, tmpV);
        }
        ids.subList(dto.sampleSize(), ids.size()).clear();
        rows.subList(dto.sampleSize(), rows.size()).clear();
      }
      float[][] data = rows.toArray(new float[0][]);
      if (Boolean.TRUE.equals(dto.sphereize())) {
        sphereize(data);
      }
      ProjectionRequest pr =
          new ProjectionRequest(
              dto.collection(),
              dto.algorithm(),
              dto.dimensions(),
              dto.sampleSize(),
              paramsFrom(dto.algorithm(), dto.dimensions(), dto.params()));
      String jobId = jobs.submit(session, pr, data, ids.toArray(new String[0]));
      res.headers().set(HeaderNames.CONTENT_TYPE, "application/json");
      res.status(Status.ACCEPTED_202)
          .send(MAPPER.writeValueAsBytes(Map.of("jobId", jobId, "n", ids.size())));
    } catch (Exception e) {
      res.status(Status.BAD_REQUEST_400).send(String.valueOf(e.getMessage()));
    }
  }

  private void events(ServerRequest req, ServerResponse res) {
    String id = req.path().pathParameters().get("id");
    ProjectionJob job = jobs.get(id);
    if (job == null) {
      res.status(Status.NOT_FOUND_404).send();
      return;
    }
    try (SseSink sink = res.sink(SseSink.TYPE)) {
      if (job.state() == ProjectionJob.State.DONE && job.result() != null) {
        try {
          sink.emit(
              SseEvent.builder()
                  .data(MAPPER.writeValueAsString(new ProjectionEvent.Done(id, job.result())))
                  .build());
        } catch (Exception ignore) {
        }
        return;
      }
      CountDownLatch done = new CountDownLatch(1);
      job.publisher()
          .subscribe(
              new Flow.Subscriber<>() {
                private Flow.Subscription sub;

                @Override
                public void onSubscribe(Flow.Subscription s) {
                  this.sub = s;
                  s.request(Long.MAX_VALUE);
                }

                @Override
                public void onNext(ProjectionEvent ev) {
                  try {
                    sink.emit(SseEvent.builder().data(MAPPER.writeValueAsString(ev)).build());
                  } catch (Exception ignore) {
                    sub.cancel();
                    done.countDown();
                  }
                }

                @Override
                public void onError(Throwable t) {
                  done.countDown();
                }

                @Override
                public void onComplete() {
                  done.countDown();
                }
              });
      try {
        done.await();
      } catch (InterruptedException ie) {
        Thread.currentThread().interrupt();
      }
    }
  }

  private void cancel(ServerRequest req, ServerResponse res) {
    String id = req.path().pathParameters().get("id");
    boolean ok = jobs.cancel(id);
    res.status(ok ? Status.OK_200 : Status.NOT_FOUND_404).send();
  }

  /**
   * Streams all vectors as raw little-endian Float32 binary for the TensorFlow Embedding Projector.
   */
  private void tensors(ServerRequest req, ServerResponse res) {
    try {
      String name = req.path().pathParameters().get("name");
      var summary = session.backend().describe(name);
      int dim = summary.dimension();
      List<float[]> rows = new ArrayList<>();
      session.backend().streamAllVectors(name, (id, v) -> rows.add(v), null);
      ByteBuffer buf = ByteBuffer.allocate(rows.size() * dim * Float.BYTES);
      buf.order(ByteOrder.LITTLE_ENDIAN);
      for (float[] vec : rows) {
        for (float v : vec) {
          buf.putFloat(v);
        }
      }
      res.headers().set(HeaderNames.CONTENT_TYPE, "application/octet-stream");
      res.send(buf.array());
    } catch (Exception e) {
      res.status(Status.INTERNAL_SERVER_ERROR_500).send(String.valueOf(e.getMessage()));
    }
  }

  /**
   * k-NN search by id. Body: {@code { "id": "doc-1", "k": 10 }}. Looks up the document's vector and
   * returns the nearest neighbours as {@code { "hits": [{ "id", "score" }, ...] }}. The Inspector
   * panel uses this; richer queries (raw vectors, filters) can be layered on later.
   */
  private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {};

  private void knnSearch(ServerRequest req, ServerResponse res) {
    try {
      String name = req.path().pathParameters().get("name");
      Map<String, Object> body = MAPPER.readValue(req.content().inputStream(), MAP_TYPE);
      String id = (String) body.get("id");
      String query = (String) body.get("query");
      int k = asInt(body.get("k"), 10);
      double mmr = asDouble(body.get("mmr"), -1.0); // >= 0 enables MMR diversity re-ranking
      Float mmrLambda = mmr >= 0 ? (float) Math.min(1.0, mmr) : null;
      List<Map<String, Object>> out;
      if (id != null && !id.isEmpty()) {
        out = searchByVector(name, id, k, mmrLambda);
      } else if (query != null && !query.isEmpty()) {
        // Free-text search: embed the query with the collection's configured model, then k-NN.
        DatasetCatalogEntry entry = catalog.byId(name).orElse(null);
        if (entry == null || entry.queryModel() == null) {
          res.status(Status.CONFLICT_409).send("text search isn't configured for this collection");
          return;
        }
        Optional<EmbeddingProvider> provider =
            providerRegistry.providerFor(entry.queryModel(), entry.queryDimensions());
        if (provider.isEmpty()) {
          res.status(Status.CONFLICT_409)
              .send(
                  "no embedding provider configured for model "
                      + entry.queryModel()
                      + " — add one on /providers");
          return;
        }
        String prefix = entry.queryPrefix() == null ? "" : entry.queryPrefix();
        float[] vec = provider.get().embed(prefix + query);
        out = searchByRawVector(name, vec, k, mmrLambda, null);
      } else {
        res.status(Status.BAD_REQUEST_400).send("missing id or query");
        return;
      }
      res.headers().set(HeaderNames.CONTENT_TYPE, "application/json");
      res.send(MAPPER.writeValueAsBytes(Map.of("hits", out)));
    } catch (IllegalArgumentException e) {
      res.status(Status.NOT_FOUND_404).send(String.valueOf(e.getMessage()));
    } catch (Exception e) {
      res.status(Status.BAD_REQUEST_400).send(String.valueOf(e.getMessage()));
    }
  }

  private static final int MMR_FETCH_MULTIPLIER = 4;

  private List<Map<String, Object>> searchByVector(String name, String id, int k, Float mmrLambda) {
    var doc = session.backend().getDocument(name, id);
    if (doc == null || doc.vector() == null) {
      throw new IllegalArgumentException("unknown id: " + id);
    }
    // Search from the document's own vector, excluding the query point itself from the results.
    return searchByRawVector(name, doc.vector(), k, mmrLambda, id);
  }

  /**
   * k-NN (optionally MMR-diversified) from a raw query vector. When {@code excludeId} is non-null
   * the matching document is dropped from the candidate pool (search-by-id excludes itself;
   * free-text search passes null so no hit is filtered).
   */
  private List<Map<String, Object>> searchByRawVector(
      String name, float[] queryVec, int k, Float mmrLambda, String excludeId) {
    int want = Math.max(1, k);
    boolean useMmr = mmrLambda != null;
    // MMR needs the candidate vectors and a bigger pool to diversify from.
    int fetch = useMmr ? want * MMR_FETCH_MULTIPLIER + 1 : want + 1;
    var spec =
        new com.integrallis.vectors.studio.core.search.SearchSpec(
            queryVec, null, fetch, null, useMmr, false, false);
    var hits = session.backend().search(name, spec);

    List<SearchHit> pool = new ArrayList<>(hits.size());
    for (var h : hits) {
      if (excludeId == null || !h.id().equals(excludeId)) {
        pool.add(h);
      }
    }

    List<SearchHit> ranked;
    if (useMmr && !pool.isEmpty()) {
      SimilarityFunction metric = parseMetric(session.backend().describe(name).metric());
      float[][] candidates = new float[pool.size()][];
      for (int i = 0; i < pool.size(); i++) {
        candidates[i] = pool.get(i).vector();
      }
      int[] selected =
          MaximalMarginalRelevance.select(
              queryVec, candidates, Math.min(want, pool.size()), mmrLambda, metric);
      ranked = new ArrayList<>(selected.length);
      for (int idx : selected) {
        ranked.add(pool.get(idx));
      }
    } else {
      ranked = pool.subList(0, Math.min(want, pool.size()));
    }

    List<Map<String, Object>> out = new ArrayList<>(ranked.size());
    for (var h : ranked) {
      out.add(Map.of("id", h.id(), "score", h.score()));
    }
    return out;
  }

  private static SimilarityFunction parseMetric(String name) {
    try {
      return SimilarityFunction.valueOf(name);
    } catch (IllegalArgumentException | NullPointerException e) {
      return SimilarityFunction.COSINE; // embeddings default; MMR ordering is robust to this
    }
  }

  /**
   * Emits id + text + every metadata column in a single TSV. Column order follows the first-seen
   * insertion order across documents; missing values are blank cells.
   */
  private void metadata(ServerRequest req, ServerResponse res) {
    try {
      String name = req.path().pathParameters().get("name");
      List<String> ids = new ArrayList<>();
      session.backend().streamAllVectors(name, (id, v) -> ids.add(id), null);
      List<com.integrallis.vectors.studio.core.search.DocumentView> docs =
          new ArrayList<>(ids.size());
      LinkedHashSet<String> columns = new LinkedHashSet<>();
      for (String id : ids) {
        var doc = session.backend().getDocument(name, id);
        docs.add(doc);
        if (doc != null && doc.metadata() != null) columns.addAll(doc.metadata().keySet());
      }
      StringBuilder tsv = new StringBuilder();
      tsv.append("id\ttext");
      for (String col : columns) tsv.append('\t').append(safeCell(col));
      tsv.append('\n');
      for (int i = 0; i < ids.size(); i++) {
        String id = ids.get(i);
        var doc = docs.get(i);
        String text = "";
        if (doc != null && doc.text() != null && !doc.text().isBlank()) {
          text = doc.text().replaceAll("[\\t\\n\\r]", " ");
          if (text.length() > 120) text = text.substring(0, 120) + "…";
        }
        tsv.append(id).append('\t').append(text);
        for (String col : columns) {
          Object v = doc != null && doc.metadata() != null ? doc.metadata().get(col) : null;
          tsv.append('\t').append(stringifyMeta(v));
        }
        tsv.append('\n');
      }
      res.headers().set(HeaderNames.CONTENT_TYPE, "text/tab-separated-values; charset=utf-8");
      res.send(tsv.toString());
    } catch (Exception e) {
      res.status(Status.INTERNAL_SERVER_ERROR_500).send(String.valueOf(e.getMessage()));
    }
  }

  private static String safeCell(String s) {
    return s == null ? "" : s.replaceAll("[\\t\\n\\r]", " ");
  }

  private static String stringifyMeta(Object v) {
    if (v == null) return "";
    if (v instanceof List<?> list) {
      StringBuilder b = new StringBuilder();
      for (int i = 0; i < list.size(); i++) {
        if (i > 0) b.append(',');
        b.append(safeCell(String.valueOf(list.get(i))));
      }
      return b.toString();
    }
    return safeCell(String.valueOf(v));
  }

  /** L2-normalises each row in place (TF Embedding Projector "Sphereize data"). */
  private static void sphereize(float[][] data) {
    for (float[] row : data) {
      double n = 0.0;
      for (float f : row) n += (double) f * f;
      n = Math.sqrt(n);
      if (n == 0.0) continue;
      float inv = (float) (1.0 / n);
      for (int i = 0; i < row.length; i++) row[i] *= inv;
    }
  }

  /** Builds typed {@link ProjectionParams} from a flat JSON map, defaulting any missing keys. */
  private static ProjectionParams paramsFrom(
      ProjectionAlgorithm a, int dims, Map<String, Object> p) {
    ProjectionParams d = defaultParams(a, dims);
    if (p == null || p.isEmpty()) return d;
    return switch (a) {
      case PCA -> {
        ProjectionParams.PcaParams base = (ProjectionParams.PcaParams) d;
        yield new ProjectionParams.PcaParams(
            asInt(p.get("components"), base.components()),
            asBool(p.get("center"), base.center()),
            asBool(p.get("whiten"), base.whiten()));
      }
      case TSNE -> {
        ProjectionParams.TsneParams base = (ProjectionParams.TsneParams) d;
        yield new ProjectionParams.TsneParams(
            asInt(p.get("perplexity"), base.perplexity()),
            asDouble(p.get("learningRate"), base.learningRate()),
            asInt(p.get("iterations"), base.iterations()),
            asLong(p.get("seed"), base.seed()));
      }
      case UMAP -> {
        ProjectionParams.UmapParams base = (ProjectionParams.UmapParams) d;
        yield new ProjectionParams.UmapParams(
            asInt(p.get("neighbors"), base.neighbors()),
            asDouble(p.get("minDist"), base.minDist()),
            asInt(p.get("iterations"), base.iterations()),
            asLong(p.get("seed"), base.seed()));
      }
    };
  }

  private static int asInt(Object v, int dflt) {
    if (v instanceof Number n) return n.intValue();
    if (v instanceof String s && !s.isEmpty()) {
      try {
        return Integer.parseInt(s);
      } catch (NumberFormatException ignore) {
      }
    }
    return dflt;
  }

  private static long asLong(Object v, long dflt) {
    if (v instanceof Number n) return n.longValue();
    if (v instanceof String s && !s.isEmpty()) {
      try {
        return Long.parseLong(s);
      } catch (NumberFormatException ignore) {
      }
    }
    return dflt;
  }

  private static double asDouble(Object v, double dflt) {
    if (v instanceof Number n) return n.doubleValue();
    if (v instanceof String s && !s.isEmpty()) {
      try {
        return Double.parseDouble(s);
      } catch (NumberFormatException ignore) {
      }
    }
    return dflt;
  }

  private static boolean asBool(Object v, boolean dflt) {
    if (v instanceof Boolean b) return b;
    if (v instanceof String s) return Boolean.parseBoolean(s);
    return dflt;
  }

  private static ProjectionParams defaultParams(ProjectionAlgorithm a, int dims) {
    return switch (a) {
      case PCA -> new ProjectionParams.PcaParams(Math.max(2, dims), true, false);
      case TSNE -> new ProjectionParams.TsneParams(15, 200.0, 1000, 42L);
      case UMAP -> new ProjectionParams.UmapParams(15, 0.1, 200, 42L);
    };
  }
}
