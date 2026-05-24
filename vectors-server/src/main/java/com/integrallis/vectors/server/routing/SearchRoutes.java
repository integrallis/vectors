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

import com.integrallis.vectors.core.filter.Filter;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.hybrid.RRFFusion;
import com.integrallis.vectors.hybrid.ScoredId;
import com.integrallis.vectors.hybrid.text.TextIndexSpi;
import com.integrallis.vectors.hybrid.text.TextSearchOutcome;
import com.integrallis.vectors.server.CollectionRegistry;
import com.integrallis.vectors.server.dto.SearchHitDto;
import com.integrallis.vectors.server.dto.SearchQuery;
import com.integrallis.vectors.server.dto.SearchResponse;
import com.integrallis.vectors.server.filter.FilterParseException;
import com.integrallis.vectors.server.filter.FilterParser;
import io.helidon.http.Status;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * HTTP route for k-NN search on a collection.
 *
 * <p>{@code POST /v1/collections/{name}/search} submits a query vector plus optional filter /
 * include flags and returns an ordered list of hits with scores, matching {@link
 * com.integrallis.vectors.db.VectorCollection#search(com.integrallis.vectors.db.SearchRequest)}.
 *
 * <p>Filters are parsed by {@link FilterParser}. Malformed filters are returned to the caller as
 * {@code 400 Bad Request}.
 */
public final class SearchRoutes implements HttpService {

  private static final Logger LOG = LoggerFactory.getLogger(SearchRoutes.class);

  private final CollectionRegistry registry;

  /**
   * @param registry shared registry of open collections
   */
  public SearchRoutes(CollectionRegistry registry) {
    this.registry = Objects.requireNonNull(registry, "registry");
  }

  @Override
  public void routing(HttpRules rules) {
    rules.post("/v1/collections/{name}/search", this::search);
  }

  private void search(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    if (!RouteSupport.validateName(name, req, res)) return;
    Optional<VectorCollection> col = registry.get(name);
    if (col.isEmpty()) {
      RouteSupport.sendProblem(res, Status.NOT_FOUND_404, "collection not found", name, req);
      return;
    }
    SearchQuery body;
    try {
      body = RouteSupport.MAPPER.readValue(req.content().inputStream(), SearchQuery.class);
    } catch (Exception e) {
      RouteSupport.sendProblem(
          res, Status.BAD_REQUEST_400, "malformed request body", e.getMessage(), req);
      return;
    }
    String validation = body.validate();
    if (validation != null) {
      RouteSupport.sendProblem(
          res, Status.BAD_REQUEST_400, "invalid search request", validation, req);
      return;
    }
    int expectedDim = col.get().config().dimension();
    if (body.queryVector().length != expectedDim) {
      RouteSupport.sendProblem(
          res,
          Status.BAD_REQUEST_400,
          "invalid search request",
          "queryVector dimension mismatch: expected "
              + expectedDim
              + ", got "
              + body.queryVector().length,
          req);
      return;
    }
    Filter filter;
    try {
      filter = FilterParser.parse(body.filter());
    } catch (FilterParseException e) {
      RouteSupport.sendProblem(res, Status.BAD_REQUEST_400, "invalid filter", e.getMessage(), req);
      return;
    }

    boolean includeVector = body.includeVectorDefault();
    boolean includeText = body.includeTextDefault();
    boolean includeMetadata = body.includeMetadataDefault();

    SearchRequest.Builder builder =
        SearchRequest.builder(body.queryVector(), body.k())
            .filter(filter)
            .includeVector(includeVector)
            .includeText(includeText)
            .includeMetadata(includeMetadata);
    if (body.efSearch() != null) {
      builder.searchListSize(body.efSearch());
    }
    if (body.minScore() != null) {
      builder.minScore(body.minScore());
    }

    try {
      long startNanos = System.nanoTime();
      SearchResult result = col.get().search(builder.build());

      // Check for hybrid search: if queryText is present and a text index exists, fuse results
      if (body.queryText() != null
          && !body.queryText().isBlank()
          && registry.getTextIndex(name).isPresent()) {
        TextIndexSpi textIndex = registry.getTextIndex(name).get();
        int retrievalK = body.k() * 3; // over-retrieve for fusion
        TextSearchOutcome textOutcome = textIndex.search(body.queryText(), retrievalK);

        // Build vector hits as ScoredId list
        List<ScoredId> vectorHits = new ArrayList<>(result.hits().size());
        Map<String, SearchResult.Hit> hitMap = new HashMap<>();
        for (SearchResult.Hit h : result.hits()) {
          vectorHits.add(new ScoredId(h.id(), h.score()));
          hitMap.put(h.id(), h);
        }

        // Build text hits as ScoredId list
        List<ScoredId> textHits = new ArrayList<>(textOutcome.size());
        for (int i = 0; i < textOutcome.size(); i++) {
          textHits.add(new ScoredId(textOutcome.ids()[i], textOutcome.scores()[i]));
        }

        // Fuse via RRF
        var fusion = new RRFFusion();
        List<ScoredId> fused = fusion.fuse(List.of(vectorHits, textHits), body.k());

        // Build response from fused results
        List<SearchHitDto> hits = new ArrayList<>(fused.size());
        for (ScoredId scored : fused) {
          SearchResult.Hit originalHit = hitMap.get(scored.id());
          if (originalHit != null) {
            hits.add(SearchHitDto.from(originalHit, includeVector, includeText, includeMetadata));
          } else {
            // Hit came from text search only — include basic info
            hits.add(new SearchHitDto(scored.id(), scored.score(), null, null, null));
          }
        }
        long elapsed = System.nanoTime() - startNanos;
        RouteSupport.sendJson(res, Status.OK_200, new SearchResponse(hits, elapsed));
      } else {
        // Standard vector-only search
        List<SearchHitDto> hits = new ArrayList<>(result.hits().size());
        for (SearchResult.Hit h : result.hits()) {
          hits.add(SearchHitDto.from(h, includeVector, includeText, includeMetadata));
        }
        RouteSupport.sendJson(
            res, Status.OK_200, new SearchResponse(hits, result.searchTimeNanos()));
      }
    } catch (IllegalArgumentException e) {
      LOG.debug("search rejected for '{}': {}", name, e.getMessage());
      RouteSupport.sendProblem(res, Status.BAD_REQUEST_400, "invalid search", e.getMessage(), req);
    }
  }
}
