package com.integrallis.vectors.server.routing;

import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.server.CollectionRegistry;
import com.integrallis.vectors.server.dto.SearchHitDto;
import com.integrallis.vectors.server.dto.SearchQuery;
import com.integrallis.vectors.server.dto.SearchResponse;
import com.integrallis.vectors.server.filter.FilterParser;
import io.helidon.http.Status;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.ArrayList;
import java.util.List;
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
 * <p>Phase 4 accepts only a {@code null} or missing filter; any present filter node falls through
 * to the {@link FilterParser} stub which throws {@link UnsupportedOperationException}, returned to
 * the caller as {@code 400 Bad Request}. Phase 5 wires the full grammar.
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
    try {
      FilterParser.parse(body.filter()); // phase 5 stub; throws on non-null filter
    } catch (UnsupportedOperationException e) {
      RouteSupport.sendProblem(
          res, Status.BAD_REQUEST_400, "unsupported filter", e.getMessage(), req);
      return;
    }

    boolean includeVector = body.includeVectorDefault();
    boolean includeText = body.includeTextDefault();
    boolean includeMetadata = body.includeMetadataDefault();

    SearchRequest.Builder builder =
        SearchRequest.builder(body.queryVector(), body.k())
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
      SearchResult result = col.get().search(builder.build());
      List<SearchHitDto> hits = new ArrayList<>(result.hits().size());
      for (SearchResult.Hit h : result.hits()) {
        hits.add(SearchHitDto.from(h, includeVector, includeText, includeMetadata));
      }
      RouteSupport.sendJson(res, Status.OK_200, new SearchResponse(hits, result.searchTimeNanos()));
    } catch (IllegalArgumentException e) {
      LOG.debug("search rejected for '{}': {}", name, e.getMessage());
      RouteSupport.sendProblem(res, Status.BAD_REQUEST_400, "invalid search", e.getMessage(), req);
    }
  }
}
