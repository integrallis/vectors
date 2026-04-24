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
import java.util.concurrent.CountDownLatch;

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
    sink.emit(SseEvent.builder().name("hello").data("{}").build());
    CountDownLatch done = new CountDownLatch(1);
    Runnable remove =
        registry.addEpochListener(
            (EpochChange change) -> {
              try {
                String payload =
                    ObjectMapperHolder.shared()
                        .writeValueAsString(Map.of("name", change.name(), "epoch", change.epoch()));
                sink.emit(SseEvent.builder().name("epoch").data(payload).build());
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
      try {
        sink.close();
      } catch (RuntimeException ignored) {
        // sink may already be closed if the client disconnected.
      }
    }
  }
}
