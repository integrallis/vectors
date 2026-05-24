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

import io.helidon.http.HeaderNames;
import io.helidon.http.Status;
import io.helidon.webserver.http.Filter;
import io.helidon.webserver.http.FilterChain;
import io.helidon.webserver.http.RoutingRequest;
import io.helidon.webserver.http.RoutingResponse;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Objects;

/** Enforces bearer-token authentication on protected HTTP routes. */
final class BearerAuthFilter implements Filter {

  private static final String BEARER = "Bearer ";

  private final byte[] expectedToken;

  BearerAuthFilter(String apiKey) {
    this.expectedToken = Objects.requireNonNull(apiKey, "apiKey").getBytes(StandardCharsets.UTF_8);
  }

  @Override
  public void filter(FilterChain chain, RoutingRequest req, RoutingResponse res) {
    if (isPublic(req.path().path())) {
      chain.proceed();
      return;
    }
    String authorization = req.headers().first(HeaderNames.AUTHORIZATION).orElse(null);
    if (isAuthorized(authorization)) {
      chain.proceed();
      return;
    }
    res.headers().set(HeaderNames.WWW_AUTHENTICATE, "Bearer");
    RouteSupport.sendProblem(
        res, Status.UNAUTHORIZED_401, "unauthorized", "missing or invalid bearer token", req);
  }

  private boolean isAuthorized(String authorization) {
    if (authorization == null || !authorization.startsWith(BEARER)) {
      return false;
    }
    byte[] actual = authorization.substring(BEARER.length()).getBytes(StandardCharsets.UTF_8);
    return MessageDigest.isEqual(expectedToken, actual);
  }

  private static boolean isPublic(String path) {
    return "/v1/health".equals(path) || "/v1/readyz".equals(path);
  }
}
