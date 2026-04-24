package com.integrallis.vectors.server.filter;

/**
 * Thrown when {@link FilterParser} encounters malformed JSON. Routes translate this into a {@code
 * 400 Bad Request} with the offending message in the {@code detail} of the problem body.
 */
public class FilterParseException extends IllegalArgumentException {

  private static final long serialVersionUID = 1L;

  public FilterParseException(String message) {
    super(message);
  }
}
