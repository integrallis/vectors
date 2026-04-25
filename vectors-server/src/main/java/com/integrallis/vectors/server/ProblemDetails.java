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
package com.integrallis.vectors.server;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * RFC 7807 {@code application/problem+json} body used for all server error responses.
 *
 * <p>Null fields are omitted on serialization so the wire payload stays compact for small cases
 * like 404s, while 400s with validation data can still carry a {@code detail}.
 *
 * @param type URI reference for the problem type (conventionally {@code about:blank} for generic
 *     HTTP-status-coded errors)
 * @param title short human-readable summary, stable across occurrences
 * @param status HTTP status code mirrored into the body
 * @param detail optional human-readable explanation for this specific occurrence
 * @param instance optional URI reference identifying the specific occurrence (e.g. request path)
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record ProblemDetails(
    String type, String title, int status, String detail, String instance) {

  /** Convenience factory for {@code about:blank} problems without a detail or instance. */
  public static ProblemDetails of(int status, String title) {
    return new ProblemDetails("about:blank", title, status, null, null);
  }

  /** Convenience factory including a detail string. */
  public static ProblemDetails of(int status, String title, String detail) {
    return new ProblemDetails("about:blank", title, status, detail, null);
  }

  /** Convenience factory including a detail string and request-path instance. */
  public static ProblemDetails of(int status, String title, String detail, String instance) {
    return new ProblemDetails("about:blank", title, status, detail, instance);
  }
}
