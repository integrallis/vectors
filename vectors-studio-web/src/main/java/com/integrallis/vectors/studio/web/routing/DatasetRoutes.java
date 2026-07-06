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
import com.integrallis.vectors.studio.core.StudioSession;
import com.integrallis.vectors.studio.core.connection.CollectionSummary;
import com.integrallis.vectors.studio.core.dataset.DatasetLoader;
import com.integrallis.vectors.studio.core.dataset.HuggingFaceRowsClient;
import com.integrallis.vectors.studio.web.dataset.DatasetCatalog;
import com.integrallis.vectors.studio.web.dataset.DatasetCatalogEntry;
import com.integrallis.vectors.studio.web.dataset.DatasetLoadEvent;
import com.integrallis.vectors.studio.web.dataset.DatasetLoadJob;
import com.integrallis.vectors.studio.web.dataset.DatasetLoadJobManager;
import com.integrallis.vectors.studio.web.view.ViewRenderer;
import io.helidon.http.HeaderNames;
import io.helidon.http.Status;
import io.helidon.http.sse.SseEvent;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import io.helidon.webserver.sse.SseSink;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Flow;
import java.util.stream.Collectors;

/**
 * Sample-dataset catalog + async loader. Serves the datasets tab page, a JSON catalog decorated
 * with per-entry loaded status, an async load endpoint that streams progress over SSE, and
 * registers the built collection into the session backend on completion. Backed by {@link
 * DatasetLoader} + {@link HuggingFaceRowsClient} (zero non-JDK transport deps).
 */
public final class DatasetRoutes implements HttpService {

  /** Server-side cap so a demo load stays quick. */
  private static final int MAX_LIMIT = 5000;

  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {};

  private final StudioSession session;
  private final ViewRenderer renderer;
  private final DatasetCatalog catalog;
  private final DatasetLoadJobManager jobs;

  public DatasetRoutes(
      StudioSession session,
      ViewRenderer renderer,
      DatasetCatalog catalog,
      DatasetLoadJobManager jobs) {
    this.session = session;
    this.renderer = renderer;
    this.catalog = catalog;
    this.jobs = jobs;
  }

  @Override
  public void routing(HttpRules rules) {
    rules
        .get("/datasets", this::page)
        .get("/api/datasets/catalog", this::catalog)
        .post("/api/datasets/load", this::load)
        .get("/api/datasets/load/{jobId}/events", this::events);
  }

  private void page(ServerRequest req, ServerResponse res) {
    renderer.render(res, "datasets.jte", Map.of("entries", catalog.entries()));
  }

  private void catalog(ServerRequest req, ServerResponse res) {
    Set<String> loaded = loadedCollectionNames();
    List<Map<String, Object>> out = new ArrayList<>(catalog.entries().size());
    for (DatasetCatalogEntry e : catalog.entries()) {
      Map<String, Object> m = new LinkedHashMap<>();
      m.put("id", e.id());
      m.put("name", e.name());
      m.put("description", e.description());
      m.put("domain", e.domain());
      m.put("model", e.model());
      m.put("dimension", e.dimension());
      m.put("metric", e.metric());
      m.put("defaultLimit", e.defaultLimit());
      m.put("loaded", loaded.contains(e.id()));
      out.add(m);
    }
    try {
      res.headers().set(HeaderNames.CONTENT_TYPE, "application/json");
      res.send(MAPPER.writeValueAsBytes(Map.of("datasets", out)));
    } catch (Exception e) {
      res.status(Status.INTERNAL_SERVER_ERROR_500).send(String.valueOf(e.getMessage()));
    }
  }

  private void load(ServerRequest req, ServerResponse res) {
    try {
      Map<String, Object> body = MAPPER.readValue(req.content().inputStream(), MAP_TYPE);
      String id = str(body.get("id"));
      if (id == null || id.isBlank()) {
        res.status(Status.BAD_REQUEST_400).send("missing 'id'");
        return;
      }
      DatasetCatalogEntry entry = catalog.byId(id).orElse(null);
      if (entry == null) {
        res.status(Status.NOT_FOUND_404).send("unknown dataset: " + id);
        return;
      }
      if (loadedCollectionNames().contains(id)) {
        res.status(Status.CONFLICT_409).send("collection already exists: " + id);
        return;
      }
      int total = Math.max(1, Math.min(entry.defaultLimit(), MAX_LIMIT));
      HuggingFaceRowsClient fetcher =
          new HuggingFaceRowsClient(entry.hfDataset(), entry.config(), entry.split());
      DatasetLoader.Config cfg =
          new DatasetLoader.Config(
              entry.vectorColumn(),
              entry.textColumn(),
              entry.idColumn(),
              total,
              parseMetric(entry.metric()));
      String jobId = jobs.submit(session, id, cfg, fetcher, total);
      res.headers().set(HeaderNames.CONTENT_TYPE, "application/json");
      res.status(Status.ACCEPTED_202).send(MAPPER.writeValueAsBytes(Map.of("jobId", jobId)));
    } catch (Exception e) {
      res.status(Status.BAD_REQUEST_400).send(String.valueOf(e.getMessage()));
    }
  }

  private void events(ServerRequest req, ServerResponse res) {
    String jobId = req.path().pathParameters().get("jobId");
    DatasetLoadJob job = jobs.get(jobId);
    if (job == null) {
      res.status(Status.NOT_FOUND_404).send();
      return;
    }
    try (SseSink sink = res.sink(SseSink.TYPE)) {
      if (job.state() != DatasetLoadJob.State.RUNNING) {
        emit(sink, job.snapshot());
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
                public void onNext(DatasetLoadEvent ev) {
                  if (!emit(sink, ev)) {
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

  private static boolean emit(SseSink sink, DatasetLoadEvent ev) {
    try {
      sink.emit(SseEvent.builder().data(MAPPER.writeValueAsString(ev)).build());
      return true;
    } catch (Exception ignore) {
      return false;
    }
  }

  private Set<String> loadedCollectionNames() {
    return session.backend().listCollections().stream()
        .map(CollectionSummary::name)
        .collect(Collectors.toSet());
  }

  private static SimilarityFunction parseMetric(String name) {
    try {
      return SimilarityFunction.valueOf(name);
    } catch (IllegalArgumentException | NullPointerException e) {
      return SimilarityFunction.COSINE;
    }
  }

  private static String str(Object v) {
    return v == null ? null : String.valueOf(v);
  }
}
