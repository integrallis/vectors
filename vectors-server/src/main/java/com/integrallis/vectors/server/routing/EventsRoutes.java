/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
package com.integrallis.vectors.server.routing;

import com.integrallis.vectors.server.CollectionRegistry;
import com.integrallis.vectors.server.CollectionRegistry.EpochChange;
import com.integrallis.vectors.server.ObjectMapperHolder;
import io.helidon.http.HeaderNames;
import io.helidon.http.Status;
import io.helidon.http.sse.SseEvent;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import io.helidon.webserver.sse.SseSink;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Route for {@code GET /v1/events} — a Server-Sent Events stream that emits one event per
 * successful commit on any collection.
 *
 * <p>Wire format (event name {@code epoch}):
 *
 * <pre>
 *   event: epoch
 *   data: {"name":"articles","epoch":42}
 * </pre>
 *
 * <p>Clients reconnect automatically per the SSE spec; there is no {@code Last-Event-Id} support
 * yet so on reconnect the client should call {@code GET /v1/collections/\u007Bname\u007D/epoch} to
 * reconcile missed deltas. The route also emits a single {@code hello} event immediately after the
 * sink is attached so clients can detect the connection without waiting for a commit.
 */
public final class EventsRoutes implements HttpService {

  private static final int EVENT_QUEUE_CAPACITY = 1024;

  private final CollectionRegistry registry;

  public EventsRoutes(CollectionRegistry registry) {
    this.registry = Objects.requireNonNull(registry, "registry");
  }

  @Override
  public void routing(HttpRules rules) {
    rules.get("/v1/events", this::stream);
  }

  private void stream(ServerRequest req, ServerResponse res) {
    res.headers().set(HeaderNames.CACHE_CONTROL, "no-cache");
    res.status(Status.OK_200);
    SseSink sink = res.sink(SseSink.TYPE);
    CountDownLatch done = new CountDownLatch(1);
    SerialSseSink serialSink = new SerialSseSink(sink, done::countDown);
    serialSink.emit(SseEvent.builder().name("hello").data("{}").build());
    Runnable remove =
        registry.addEpochListener(
            (EpochChange change) -> {
              try {
                String payload =
                    ObjectMapperHolder.shared()
                        .writeValueAsString(Map.of("name", change.name(), "epoch", change.epoch()));
                if (!serialSink.emit(SseEvent.builder().name("epoch").data(payload).build())) {
                  done.countDown();
                }
              } catch (RuntimeException | com.fasterxml.jackson.core.JsonProcessingException ex) {
                // Client disconnected, sink closed, or serialization error — clean up below.
                done.countDown();
              }
            });
    try {
      // Block the virtual thread until the sink closes (client disconnect or server shutdown).
      // The latch is counted down when the sink reports an error (client disconnect).
      done.await();
    } catch (InterruptedException ignored) {
      Thread.currentThread().interrupt();
    } finally {
      remove.run();
      serialSink.close();
    }
  }

  private static final class SerialSseSink implements AutoCloseable {

    private final SseSink sink;
    private final BlockingQueue<SseEvent> queue = new ArrayBlockingQueue<>(EVENT_QUEUE_CAPACITY);
    private final AtomicBoolean closed = new AtomicBoolean();
    private final Runnable onFailure;
    private final Thread worker;

    private SerialSseSink(SseSink sink, Runnable onFailure) {
      this.sink = Objects.requireNonNull(sink, "sink");
      this.onFailure = Objects.requireNonNull(onFailure, "onFailure");
      this.worker = Thread.ofVirtual().name("vectors-sse-sink").start(this::drain);
    }

    private boolean emit(SseEvent event) {
      Objects.requireNonNull(event, "event");
      if (closed.get()) {
        return false;
      }
      if (queue.offer(event)) {
        return true;
      }
      onFailure.run();
      close();
      return false;
    }

    private void drain() {
      try {
        while (!closed.get()) {
          SseEvent event = queue.take();
          sink.emit(event);
        }
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      } catch (RuntimeException e) {
        closed.set(true);
        onFailure.run();
      }
    }

    @Override
    public void close() {
      if (closed.compareAndSet(false, true)) {
        worker.interrupt();
      }
      try {
        sink.close();
      } catch (RuntimeException ignored) {
        // sink may already be closed if the client disconnected.
      }
    }
  }
}
