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

import com.integrallis.vectors.studio.core.StudioSession;
import com.integrallis.vectors.studio.core.search.SearchHit;
import com.integrallis.vectors.studio.core.search.SearchSpec;
import com.integrallis.vectors.studio.web.view.ViewRenderer;
import io.helidon.http.Status;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.List;
import java.util.Map;

/** Vector / hybrid search HTMX route — POST returns a hits-list fragment. */
public final class SearchRoutes implements HttpService {

  private final StudioSession session;
  private final ViewRenderer renderer;

  public SearchRoutes(StudioSession session, ViewRenderer renderer) {
    this.session = session;
    this.renderer = renderer;
  }

  @Override
  public void routing(HttpRules rules) {
    rules.post("/collections/{name}/search", this::execute);
  }

  private void execute(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    var summary = session.backend().describe(name);
    String body = req.content().as(String.class);
    Map<String, String> form = parseForm(body);
    String text = form.get("query");
    int k = parseIntOr(form.get("k"), 10);
    SearchSpec spec =
        new SearchSpec(zeroVector(summary.dimension(), text), text, k, null, false, true, true);
    List<SearchHit> hits;
    try {
      hits = session.backend().search(name, spec);
    } catch (RuntimeException e) {
      hits = List.of();
    }
    renderer.renderFragment(res, Status.OK_200, "partials/hitsList.jte", Map.of("hits", hits));
  }

  private static float[] zeroVector(int dim, String seed) {
    float[] v = new float[dim];
    if (seed == null || seed.isEmpty()) return v;
    int h = seed.hashCode();
    for (int i = 0; i < dim; i++) {
      v[i] = ((h >>> (i % 32)) & 1) == 0 ? -1.0f : 1.0f;
    }
    double n = 0.0;
    for (float f : v) n += f * f;
    n = Math.sqrt(n);
    float scale = (float) (n == 0 ? 1.0 : 1.0 / n);
    for (int i = 0; i < dim; i++) v[i] *= scale;
    return v;
  }

  private static Map<String, String> parseForm(String body) {
    java.util.HashMap<String, String> out = new java.util.HashMap<>();
    if (body == null) return out;
    for (String pair : body.split("&")) {
      int eq = pair.indexOf('=');
      if (eq <= 0) continue;
      out.put(
          java.net.URLDecoder.decode(
              pair.substring(0, eq), java.nio.charset.StandardCharsets.UTF_8),
          java.net.URLDecoder.decode(
              pair.substring(eq + 1), java.nio.charset.StandardCharsets.UTF_8));
    }
    return out;
  }

  private static int parseIntOr(String s, int dflt) {
    if (s == null || s.isEmpty()) return dflt;
    try {
      return Integer.parseInt(s);
    } catch (NumberFormatException e) {
      return dflt;
    }
  }
}
