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
package com.integrallis.vectors.studio.web.view;

import gg.jte.TemplateEngine;
import gg.jte.output.StringOutput;
import io.helidon.http.HeaderNames;
import io.helidon.http.Status;
import io.helidon.webserver.http.ServerResponse;
import java.util.Map;
import java.util.Objects;

/** Renders a JTE template by name into an HTTP response with {@code text/html} content type. */
public final class ViewRenderer {

  private final TemplateEngine engine;

  public ViewRenderer(TemplateEngine engine) {
    this.engine = Objects.requireNonNull(engine, "engine");
  }

  /** Renders {@code template} with {@code params} and writes it to {@code response} as 200 OK. */
  public void render(ServerResponse response, String template, Map<String, Object> params) {
    StringOutput out = new StringOutput();
    engine.render(template, params, out);
    response.headers().set(HeaderNames.CONTENT_TYPE, "text/html; charset=utf-8");
    response.status(Status.OK_200).send(out.toString());
  }

  /** Renders an HTML fragment with the given status. */
  public void renderFragment(
      ServerResponse response, Status status, String template, Map<String, Object> params) {
    StringOutput out = new StringOutput();
    engine.render(template, params, out);
    response.headers().set(HeaderNames.CONTENT_TYPE, "text/html; charset=utf-8");
    response.status(status).send(out.toString());
  }
}
