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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.vectors.studio.web.embed.ProviderRegistry;
import com.integrallis.vectors.studio.web.embed.ProviderStatus;
import com.integrallis.vectors.studio.web.view.ViewRenderer;
import io.helidon.http.HeaderNames;
import io.helidon.http.Status;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Embedding-provider settings page + JSON status endpoint. Both surfaces are redacted: they report
 * only whether an API key is present, never the key value.
 */
public final class ProvidersRoutes implements HttpService {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  private final ViewRenderer renderer;
  private final ProviderRegistry registry;

  public ProvidersRoutes(ViewRenderer renderer, ProviderRegistry registry) {
    this.renderer = renderer;
    this.registry = registry;
  }

  @Override
  public void routing(HttpRules rules) {
    rules.get("/providers", this::page).get("/api/providers", this::json);
  }

  private void page(ServerRequest req, ServerResponse res) {
    renderer.render(
        res,
        "providers.jte",
        Map.of("providers", registry.statuses(), "overrideEnv", ProviderRegistry.OVERRIDE_ENV));
  }

  private void json(ServerRequest req, ServerResponse res) {
    List<Map<String, Object>> out = new ArrayList<>();
    for (ProviderStatus s : registry.statuses()) {
      Map<String, Object> m = new LinkedHashMap<>();
      m.put("id", s.id());
      m.put("type", s.type());
      m.put("models", s.models());
      m.put("keyPresent", s.keyPresent());
      out.add(m);
    }
    try {
      res.headers().set(HeaderNames.CONTENT_TYPE, "application/json");
      res.send(MAPPER.writeValueAsBytes(Map.of("providers", out)));
    } catch (Exception e) {
      res.status(Status.INTERNAL_SERVER_ERROR_500).send(String.valueOf(e.getMessage()));
    }
  }
}
