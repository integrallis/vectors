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

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.server.CollectionRegistry;
import com.integrallis.vectors.server.dto.DocumentDto;
import com.integrallis.vectors.server.dto.DocumentPageDto;
import com.integrallis.vectors.server.dto.MetadataCodec;
import com.integrallis.vectors.server.dto.SampleResponseDto;
import com.integrallis.vectors.server.dto.VectorsBatchRequestDto;
import com.integrallis.vectors.server.dto.VectorsBatchResponseDto;
import io.helidon.http.Status;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.random.RandomGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Bulk read endpoints used by Studio and similar exploration tools.
 *
 * <ul>
 *   <li>{@code GET /v1/collections/{name}/documents?offset=&limit=&includeVectors=} — paginated
 *       preview of live documents
 *   <li>{@code POST /v1/collections/{name}/vectors-batch} — high-throughput vector batch fetch by
 *       id
 *   <li>{@code GET /v1/collections/{name}/sample?n=&includeMetadata=} — uniform random sample
 * </ul>
 */
public final class BulkRoutes implements HttpService {

  private static final Logger LOG = LoggerFactory.getLogger(BulkRoutes.class);

  /** Hard upper bound on a single page or sample to prevent runaway memory allocations. */
  private static final int MAX_PAGE = 10_000;

  private final CollectionRegistry registry;

  public BulkRoutes(CollectionRegistry registry) {
    this.registry = Objects.requireNonNull(registry, "registry");
  }

  @Override
  public void routing(HttpRules rules) {
    rules
        .get("/v1/collections/{name}/documents", this::listDocuments)
        .post("/v1/collections/{name}/vectors-batch", this::vectorsBatch)
        .get("/v1/collections/{name}/sample", this::sample);
  }

  private void listDocuments(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    if (!RouteSupport.validateName(name, req, res)) return;
    Optional<VectorCollection> col = registry.get(name);
    if (col.isEmpty()) {
      RouteSupport.sendProblem(res, Status.NOT_FOUND_404, "collection not found", name, req);
      return;
    }
    int offset = parseInt(req, "offset", 0);
    int limit = parseInt(req, "limit", 100);
    boolean includeVectors = parseBool(req, "includeVectors", false);
    if (offset < 0 || limit <= 0 || limit > MAX_PAGE) {
      RouteSupport.sendProblem(
          res,
          Status.BAD_REQUEST_400,
          "invalid pagination",
          "offset>=0, 0<limit<=" + MAX_PAGE,
          req);
      return;
    }
    List<Document> all = col.get().documents();
    int from = Math.min(offset, all.size());
    int to = Math.min(from + limit, all.size());
    List<DocumentDto> page = new ArrayList<>(to - from);
    for (int i = from; i < to; i++) {
      Document d = all.get(i);
      page.add(
          new DocumentDto(
              d.id(),
              includeVectors ? d.vector() : null,
              d.text(),
              MetadataCodec.toJson(d.metadata()),
              null));
    }
    RouteSupport.sendJson(res, Status.OK_200, new DocumentPageDto(page, all.size()));
  }

  private void vectorsBatch(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    if (!RouteSupport.validateName(name, req, res)) return;
    Optional<VectorCollection> col = registry.get(name);
    if (col.isEmpty()) {
      RouteSupport.sendProblem(res, Status.NOT_FOUND_404, "collection not found", name, req);
      return;
    }
    VectorsBatchRequestDto body;
    try {
      body =
          RouteSupport.MAPPER.readValue(req.content().inputStream(), VectorsBatchRequestDto.class);
    } catch (Exception e) {
      // Don't echo Jackson's parser detail to the client — it leaks source positions and parser
      // internals. The actual exception is logged for operators.
      LOG.debug("rejected malformed vectors-batch body: {}", e.toString());
      RouteSupport.sendProblem(
          res,
          Status.BAD_REQUEST_400,
          "malformed request body",
          "request body is not valid JSON or does not match the expected schema",
          req);
      return;
    }
    if (body.ids().size() > MAX_PAGE) {
      RouteSupport.sendProblem(
          res,
          Status.BAD_REQUEST_400,
          "batch too large",
          "max " + MAX_PAGE + " ids per request, got " + body.ids().size(),
          req);
      return;
    }
    VectorCollection c = col.get();
    List<float[]> hits = new ArrayList<>(body.ids().size());
    List<String> missing = new ArrayList<>();
    for (String id : body.ids()) {
      Document d = c.get(id);
      if (d == null || d.vector() == null) {
        missing.add(id);
      } else {
        hits.add(d.vector());
      }
    }
    float[][] mat = hits.toArray(new float[0][]);
    RouteSupport.sendJson(res, Status.OK_200, new VectorsBatchResponseDto(mat, missing));
  }

  private void sample(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    if (!RouteSupport.validateName(name, req, res)) return;
    Optional<VectorCollection> col = registry.get(name);
    if (col.isEmpty()) {
      RouteSupport.sendProblem(res, Status.NOT_FOUND_404, "collection not found", name, req);
      return;
    }
    int n = parseInt(req, "n", 1000);
    boolean includeMetadata = parseBool(req, "includeMetadata", false);
    if (n <= 0 || n > MAX_PAGE) {
      RouteSupport.sendProblem(
          res, Status.BAD_REQUEST_400, "invalid sample size", "0<n<=" + MAX_PAGE, req);
      return;
    }
    List<Document> all = col.get().documents();
    int total = all.size();
    int take = Math.min(n, total);
    Set<Integer> picked = reservoirIndices(total, take);
    String[] ids = new String[take];
    float[][] vecs = new float[take][];
    List<com.fasterxml.jackson.databind.JsonNode> meta =
        includeMetadata ? new ArrayList<>(take) : null;
    int i = 0;
    for (int idx : picked) {
      Document d = all.get(idx);
      ids[i] = d.id();
      vecs[i] = d.vector();
      if (meta != null) meta.add(MetadataCodec.toJson(d.metadata()));
      i++;
    }
    RouteSupport.sendJson(res, Status.OK_200, new SampleResponseDto(ids, vecs, meta));
  }

  static Set<Integer> reservoirIndices(int total, int take) {
    return reservoirIndices(total, take, ThreadLocalRandom.current());
  }

  static Set<Integer> reservoirIndices(int total, int take, RandomGenerator rng) {
    Objects.requireNonNull(rng, "rng");
    if (take >= total) {
      Set<Integer> all = new HashSet<>(total);
      for (int i = 0; i < total; i++) all.add(i);
      return all;
    }
    int[] reservoir = new int[take];
    for (int i = 0; i < take; i++) {
      reservoir[i] = i;
    }
    for (int i = take; i < total; i++) {
      int slot = rng.nextInt(i + 1);
      if (slot < take) {
        reservoir[slot] = i;
      }
    }
    Set<Integer> picked = new HashSet<>(take * 2);
    for (int idx : reservoir) picked.add(idx);
    return picked;
  }

  private static int parseInt(ServerRequest req, String key, int dflt) {
    return req.query().first(key).asOptional().map(Integer::parseInt).orElse(dflt);
  }

  private static boolean parseBool(ServerRequest req, String key, boolean dflt) {
    return req.query().first(key).asOptional().map(Boolean::parseBoolean).orElse(dflt);
  }
}
