package com.integrallis.vectors.server.filter;

import com.fasterxml.jackson.databind.JsonNode;
import com.integrallis.vectors.db.filter.Filter;
import com.integrallis.vectors.db.filter.Filters;
import java.util.ArrayList;
import java.util.List;

/**
 * Parses a JSON filter body into a vectors-db {@link Filter} AST.
 *
 * <p>Grammar (evaluated case-sensitively):
 *
 * <pre>
 *   filter      := null | object
 *   object      := field-pred | logical | match-all
 *   match-all   := { }
 *   field-pred  := { "field": STR, &lt;op&gt;: VALUE } (+ optional second op for range)
 *                    op ∈ { eq, ne, in, nin, gt, gte, lt, lte }
 *   logical     := { "and": [filter, ...] } | { "or": [filter, ...] } | { "not": filter }
 * </pre>
 *
 * <p>Field-predicate composition rules:
 *
 * <ul>
 *   <li>{@code eq}/{@code ne} accept any scalar literal (string, number, boolean).
 *   <li>{@code in}/{@code nin} accept a non-empty array of strings <i>or</i> numbers (uniform
 *       type).
 *   <li>{@code gt}/{@code gte}/{@code lt}/{@code lte} accept a numeric literal only. A single field
 *       predicate may carry one lower bound ({@code gt}/{@code gte}) and one upper bound ({@code
 *       lt}/{@code lte}); combining more than one of each family is rejected.
 *   <li>Range and equality ops may not be combined on the same predicate object.
 * </ul>
 *
 * <p>Malformed input produces a {@link FilterParseException} (subclass of {@link
 * IllegalArgumentException}) with an English description of the offending node.
 */
public final class FilterParser {

  private FilterParser() {}

  /** Null or missing → {@link Filters#all()}. */
  public static Filter parse(JsonNode node) {
    if (node == null || node.isNull() || node.isMissingNode()) {
      return Filters.all();
    }
    if (!node.isObject()) {
      throw new FilterParseException("filter must be a JSON object or null");
    }
    if (node.size() == 0) {
      return Filters.all();
    }
    if (node.has("and")) {
      return new Filter.And(parseChildren(node.get("and"), "and"));
    }
    if (node.has("or")) {
      return new Filter.Or(parseChildren(node.get("or"), "or"));
    }
    if (node.has("not")) {
      return new Filter.Not(parse(node.get("not")));
    }
    return parseFieldPredicate(node);
  }

  private static List<Filter> parseChildren(JsonNode arr, String key) {
    if (arr == null || !arr.isArray() || arr.isEmpty()) {
      throw new FilterParseException("'" + key + "' must be a non-empty array of filter objects");
    }
    List<Filter> out = new ArrayList<>(arr.size());
    for (JsonNode child : arr) {
      out.add(parse(child));
    }
    return out;
  }

  private static final int MAX_FIELD_NAME_LENGTH = 256;

  private static Filter parseFieldPredicate(JsonNode node) {
    JsonNode f = node.get("field");
    if (f == null || !f.isTextual() || f.asText().isEmpty()) {
      throw new FilterParseException("predicate missing required 'field' string");
    }
    String field = f.asText();
    if (field.length() > MAX_FIELD_NAME_LENGTH) {
      throw new FilterParseException(
          "field name exceeds maximum length of " + MAX_FIELD_NAME_LENGTH + " characters");
    }

    if (node.has("eq")) {
      return new Filter.Eq(field, scalar(node.get("eq"), "eq"));
    }
    if (node.has("ne")) {
      return new Filter.Not(new Filter.Eq(field, scalar(node.get("ne"), "ne")));
    }
    if (node.has("in")) {
      return parseIn(field, node.get("in"), false);
    }
    if (node.has("nin")) {
      return new Filter.Not(parseIn(field, node.get("nin"), true));
    }
    // range composition: any of gt/gte + lt/lte may appear together
    Double lower = null;
    boolean lowerInc = false;
    Double upper = null;
    boolean upperInc = false;
    if (node.has("gte")) {
      lower = number(node.get("gte"), "gte");
      lowerInc = true;
    } else if (node.has("gt")) {
      lower = number(node.get("gt"), "gt");
    }
    if (node.has("lte")) {
      upper = number(node.get("lte"), "lte");
      upperInc = true;
    } else if (node.has("lt")) {
      upper = number(node.get("lt"), "lt");
    }
    if (lower == null && upper == null) {
      throw new FilterParseException(
          "predicate for '" + field + "' has no operator (expected eq/ne/in/nin/gt/gte/lt/lte)");
    }
    return new Filter.NumericRange(field, lower, lowerInc, upper, upperInc);
  }

  /** Returns either a String, Long, Double, or Boolean, matching {@link Filters#eq} overloads. */
  private static Object scalar(JsonNode v, String op) {
    if (v.isTextual()) return v.asText();
    if (v.isBoolean()) return v.asBoolean();
    if (v.isIntegralNumber()) return v.asLong();
    if (v.isNumber()) return v.asDouble();
    throw new FilterParseException(
        "'" + op + "' value must be a string, number, or boolean (got " + v.getNodeType() + ")");
  }

  private static double number(JsonNode v, String op) {
    if (!v.isNumber()) {
      throw new FilterParseException(
          "'" + op + "' value must be numeric (got " + v.getNodeType() + ")");
    }
    return v.asDouble();
  }

  private static Filter parseIn(String field, JsonNode arr, boolean negated) {
    if (arr == null || !arr.isArray() || arr.isEmpty()) {
      throw new FilterParseException(
          "'" + (negated ? "nin" : "in") + "' for '" + field + "' requires a non-empty array");
    }
    boolean numeric = arr.get(0).isNumber();
    boolean textual = arr.get(0).isTextual();
    if (!numeric && !textual) {
      throw new FilterParseException(
          "'" + (negated ? "nin" : "in") + "' elements must be strings or numbers");
    }
    List<Object> values = new ArrayList<>(arr.size());
    for (JsonNode el : arr) {
      if (numeric) {
        if (!el.isNumber()) {
          throw new FilterParseException(
              "'" + (negated ? "nin" : "in") + "' array must be uniformly numeric");
        }
        values.add(el.isIntegralNumber() ? (Object) el.asLong() : el.asDouble());
      } else {
        if (!el.isTextual()) {
          throw new FilterParseException(
              "'" + (negated ? "nin" : "in") + "' array must be uniformly string");
        }
        values.add(el.asText());
      }
    }
    return new Filter.In(field, values);
  }
}
