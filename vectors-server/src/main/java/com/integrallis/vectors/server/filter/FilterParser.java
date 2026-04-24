package com.integrallis.vectors.server.filter;

import com.fasterxml.jackson.databind.JsonNode;

/**
 * Parser for JSON filter bodies submitted to {@code POST /v1/collections/{name}/search}.
 *
 * <p>Phase 2 placeholder: supports only a {@code null}/missing filter, returning {@link
 * #MATCH_ALL}. Subsequent phases implement the full {@code equals / notEquals / in / range / and /
 * or / not} grammar on top of {@link com.integrallis.vectors.db.MetadataFilter}.
 */
public final class FilterParser {

  /** Sentinel returned for a null or missing filter node. */
  public static final Object MATCH_ALL = new Object();

  private FilterParser() {}

  /**
   * Parses a filter {@link JsonNode}, or returns {@link #MATCH_ALL} if the node is {@code null} or
   * a JSON null.
   *
   * @param node filter JSON, may be {@code null}
   * @return parsed filter, or {@link #MATCH_ALL}
   * @throws UnsupportedOperationException when the node is non-null (deferred to Phase 5)
   */
  public static Object parse(JsonNode node) {
    if (node == null || node.isNull()) {
      return MATCH_ALL;
    }
    throw new UnsupportedOperationException("filter parsing is deferred to vectors-server phase 5");
  }
}
