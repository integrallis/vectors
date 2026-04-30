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
package com.integrallis.vectors.server.client;

/** Thrown when the vectors-server returns a non-2xx HTTP response. */
public final class VectorsServerException extends RuntimeException {

  @java.io.Serial private static final long serialVersionUID = 1L;

  private final int statusCode;
  private final String body;

  public VectorsServerException(int statusCode, String body) {
    super("HTTP " + statusCode + ": " + body);
    this.statusCode = statusCode;
    this.body = body;
  }

  public int statusCode() {
    return statusCode;
  }

  public String body() {
    return body;
  }
}
