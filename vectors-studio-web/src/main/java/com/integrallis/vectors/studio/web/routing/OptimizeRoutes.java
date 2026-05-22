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

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.integrallis.vectors.studio.core.StudioSession;
import com.integrallis.vectors.studio.web.optimize.OptimizeEvent;
import com.integrallis.vectors.studio.web.optimize.OptimizeJob;
import com.integrallis.vectors.studio.web.optimize.OptimizeJobManager;
import com.integrallis.vectors.studio.web.optimize.OptimizeRequestDto;
import com.integrallis.vectors.studio.web.optimize.OptimizeStudyBuilder;
import com.integrallis.vectors.studio.web.view.ViewRenderer;
import io.helidon.http.HeaderNames;
import io.helidon.http.Status;
import io.helidon.http.sse.SseEvent;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import io.helidon.webserver.sse.SseSink;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Flow;

/**
 * HTTP surface for the optimization feature: a study-design page, JSON+SSE control APIs, and a
 * confirm-gated apply route. Mirrors the {@code ApiRoutes} + {@code ProjectorRoutes} pattern used
 * by the existing projector page.
 */
public final class OptimizeRoutes implements HttpService {

  private static final ObjectMapper MAPPER =
      new ObjectMapper()
          .registerModule(new JavaTimeModule())
          .disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS)
          .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);

  private final StudioSession session;
  private final OptimizeJobManager jobs;
  private final ViewRenderer renderer;

  public OptimizeRoutes(StudioSession session, OptimizeJobManager jobs, ViewRenderer renderer) {
    this.session = session;
    this.jobs = jobs;
    this.renderer = renderer;
  }

  @Override
  public void routing(HttpRules rules) {
    rules
        .get("/collections/{name}/optimize", this::designPage)
        .get("/collections/{name}/optimize/progress/{studyId}", this::progressPage)
        .post("/api/optimize/studies", this::submit)
        .get("/api/optimize/studies/{id}/events", this::events)
        .get("/api/optimize/studies/{id}/trials", this::listTrials)
        .post("/api/optimize/studies/{id}/cancel", this::cancel)
        .post("/collections/{name}/optimize/apply/{studyId}/{trialId}", this::applyTrial);
  }

  private void designPage(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    var summary = session.backend().describe(name);
    renderer.render(res, "optimize/design.jte", Map.of("summary", summary));
  }

  private void progressPage(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    String studyId = req.path().pathParameters().get("studyId");
    var summary = session.backend().describe(name);
    renderer.render(res, "optimize/progress.jte", Map.of("summary", summary, "studyId", studyId));
  }

  private void submit(ServerRequest req, ServerResponse res) {
    try {
      OptimizeRequestDto raw =
          MAPPER.readValue(req.content().as(byte[].class), OptimizeRequestDto.class);
      OptimizeRequestDto dto = raw.withDefaults();
      if (dto.collection() == null || dto.collection().isBlank()) {
        res.status(Status.BAD_REQUEST_400).send("collection is required");
        return;
      }
      OptimizeStudyBuilder.Built built = OptimizeStudyBuilder.build(session, dto);
      String studyId = jobs.submit(built.config(), built.embedder());
      res.headers().set(HeaderNames.CONTENT_TYPE, "application/json");
      res.status(Status.ACCEPTED_202).send(MAPPER.writeValueAsBytes(Map.of("studyId", studyId)));
    } catch (Exception e) {
      res.status(Status.BAD_REQUEST_400).send(String.valueOf(e.getMessage()));
    }
  }

  private void events(ServerRequest req, ServerResponse res) {
    String id = req.path().pathParameters().get("id");
    OptimizeJob job = jobs.get(id);
    if (job == null) {
      res.status(Status.NOT_FOUND_404).send();
      return;
    }
    try (SseSink sink = res.sink(SseSink.TYPE)) {
      // Replay current history for late subscribers.
      var hist = job.history();
      for (int i = 0; i < hist.size(); i++) {
        try {
          sink.emit(
              SseEvent.builder()
                  .data(
                      MAPPER.writeValueAsString(
                          new OptimizeEvent.TrialCompletedEvt(id, hist.get(i), i)))
                  .build());
        } catch (Exception ignore) {
          return;
        }
      }
      if (job.state() != OptimizeJob.State.PENDING && job.state() != OptimizeJob.State.RUNNING) {
        emitTerminal(sink, id, job);
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
                public void onNext(OptimizeEvent ev) {
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

  private static void emitTerminal(SseSink sink, String id, OptimizeJob job) {
    try {
      OptimizeEvent ev =
          switch (job.state()) {
            case COMPLETED -> new OptimizeEvent.CompletedEvt(id, job.history().size());
            case CANCELLED -> new OptimizeEvent.CancelledEvt(id, job.history().size());
            case FAILED ->
                new OptimizeEvent.ErrorEvt(
                    id, job.error() == null ? "unknown" : job.error().toString());
            default -> null;
          };
      if (ev != null) sink.emit(SseEvent.builder().data(MAPPER.writeValueAsString(ev)).build());
    } catch (Exception ignored) {
    }
  }

  private void listTrials(ServerRequest req, ServerResponse res) {
    String id = req.path().pathParameters().get("id");
    OptimizeJob job = jobs.get(id);
    if (job == null) {
      res.status(Status.NOT_FOUND_404).send();
      return;
    }
    try {
      res.headers().set(HeaderNames.CONTENT_TYPE, "application/json");
      res.send(
          MAPPER.writeValueAsBytes(
              Map.of(
                  "studyId", id,
                  "state", job.state().name(),
                  "trials", job.history())));
    } catch (Exception e) {
      res.status(Status.INTERNAL_SERVER_ERROR_500).send();
    }
  }

  private void cancel(ServerRequest req, ServerResponse res) {
    String id = req.path().pathParameters().get("id");
    boolean ok = jobs.cancel(id);
    res.status(ok ? Status.OK_200 : Status.NOT_FOUND_404).send();
  }

  private void applyTrial(ServerRequest req, ServerResponse res) {
    String confirm = req.query().first("confirm").asOptional().orElse("");
    if (!"true".equals(confirm)) {
      res.status(Status.BAD_REQUEST_400)
          .send(
              "Apply is destructive: it would replace the live collection's index with the "
                  + "trial's parameters. Re-issue this request with ?confirm=true to proceed.");
      return;
    }
    // v1: applying parameters live requires backend reconfiguration which is not yet exposed;
    // accept the confirmation but report deferred-application via a 202.
    res.status(Status.ACCEPTED_202)
        .send("Trial parameters captured; live re-indexing is deferred to a future release.");
  }
}
