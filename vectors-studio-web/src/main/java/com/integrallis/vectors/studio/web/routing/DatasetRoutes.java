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

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.studio.core.StudioSession;
import com.integrallis.vectors.studio.core.dataset.DatasetLoader;
import com.integrallis.vectors.studio.core.dataset.HuggingFaceRowsClient;
import io.helidon.http.HeaderNames;
import io.helidon.http.Status;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.Map;

/**
 * JSON API for loading a sample dataset from the HuggingFace datasets-server into a new in-process
 * collection. Backed by {@link DatasetLoader} + {@link HuggingFaceRowsClient} (zero non-JDK
 * transport deps).
 */
public final class DatasetRoutes implements HttpService {

  /** Server-side cap so a demo load stays quick. */
  private static final int MAX_LIMIT = 5000;

  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {};

  private final StudioSession session;

  public DatasetRoutes(StudioSession session) {
    this.session = session;
  }

  @Override
  public void routing(HttpRules rules) {
    rules.post("/api/datasets/load", this::load);
  }

  private void load(ServerRequest req, ServerResponse res) {
    try {
      Map<String, Object> body = MAPPER.readValue(req.content().inputStream(), MAP_TYPE);
      String dataset = str(body.get("dataset"));
      String name = str(body.get("name"));
      String vectorColumn = str(body.get("vectorColumn"));
      String textColumn = str(body.get("textColumn"));
      String idColumn = str(body.get("idColumn"));
      String config = str(body.get("config"));
      String split = str(body.get("split"));
      int limit = asInt(body.get("limit"), 2000);

      if (dataset == null || dataset.isBlank()) {
        throw new IllegalArgumentException("missing 'dataset'");
      }
      if (name == null || name.isBlank()) {
        throw new IllegalArgumentException("missing 'name'");
      }
      if (vectorColumn == null || vectorColumn.isBlank()) {
        throw new IllegalArgumentException("missing 'vectorColumn'");
      }
      limit = Math.max(1, Math.min(limit, MAX_LIMIT));

      HuggingFaceRowsClient fetcher = new HuggingFaceRowsClient(dataset, config, split);
      DatasetLoader.Config cfg =
          new DatasetLoader.Config(
              vectorColumn, textColumn, idColumn, limit, SimilarityFunction.COSINE);
      VectorCollection collection = DatasetLoader.load(cfg, fetcher);
      session.backend().addCollection(name, collection);

      res.headers().set(HeaderNames.CONTENT_TYPE, "application/json");
      res.send(MAPPER.writeValueAsBytes(Map.of("name", name, "size", collection.size())));
    } catch (Exception e) {
      res.status(Status.BAD_REQUEST_400).send(String.valueOf(e.getMessage()));
    }
  }

  private static String str(Object v) {
    return v == null ? null : String.valueOf(v);
  }

  private static int asInt(Object v, int dflt) {
    if (v instanceof Number n) return n.intValue();
    if (v instanceof String s && !s.isEmpty()) {
      try {
        return Integer.parseInt(s);
      } catch (NumberFormatException ignore) {
        // fall through to default
      }
    }
    return dflt;
  }
}
