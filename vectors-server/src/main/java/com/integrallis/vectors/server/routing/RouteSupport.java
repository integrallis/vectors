package com.integrallis.vectors.server.routing;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.vectors.server.ObjectMapperHolder;
import com.integrallis.vectors.server.ProblemDetails;
import io.helidon.http.HeaderNames;
import io.helidon.http.Status;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Shared helpers for the HTTP route services: writing JSON bodies, RFC 7807 problem bodies, and
 * forwarding {@link IllegalArgumentException} / {@link UnsupportedOperationException} as 400s.
 */
final class RouteSupport {

  private static final Logger LOG = LoggerFactory.getLogger(RouteSupport.class);
  static final String JSON = "application/json";
  static final String PROBLEM = "application/problem+json";
  static final ObjectMapper MAPPER = ObjectMapperHolder.shared();

  private RouteSupport() {}

  static void sendJson(ServerResponse res, Status status, Object body) {
    try {
      String json = MAPPER.writeValueAsString(body);
      res.headers().set(HeaderNames.CONTENT_TYPE, JSON);
      res.status(status).send(json);
    } catch (Exception e) {
      LOG.error("failed to serialize response body", e);
      res.status(Status.INTERNAL_SERVER_ERROR_500).send();
    }
  }

  static void sendProblem(
      ServerResponse res, Status status, String title, String detail, ServerRequest req) {
    ProblemDetails pd = ProblemDetails.of(status.code(), title, detail, req.path().path());
    try {
      res.headers().set(HeaderNames.CONTENT_TYPE, PROBLEM);
      res.status(status).send(MAPPER.writeValueAsString(pd));
    } catch (Exception e) {
      LOG.error("failed to serialize problem body", e);
      res.status(Status.INTERNAL_SERVER_ERROR_500).send();
    }
  }
}
