package com.integrallis.vectors.spring.ai;

import com.integrallis.vectors.db.filter.Filter;
import com.integrallis.vectors.db.filter.Filters;
import java.util.List;
import org.springframework.ai.vectorstore.filter.Filter.Expression;
import org.springframework.ai.vectorstore.filter.Filter.Group;
import org.springframework.ai.vectorstore.filter.Filter.Key;
import org.springframework.ai.vectorstore.filter.Filter.Operand;
import org.springframework.ai.vectorstore.filter.Filter.Value;

/**
 * Converts Spring AI {@link Expression} filter AST to java-vectors {@link Filter} AST.
 *
 * <p>This class is stateless; all methods are static.
 */
final class FilterConverter {

  private FilterConverter() {}

  /**
   * Converts a Spring AI filter operand to a java-vectors filter.
   *
   * @param operand the Spring AI filter operand (Expression, Group, or null)
   * @return the java-vectors filter, never null ({@link Filter.All} for null input)
   */
  static Filter convert(Operand operand) {
    if (operand == null) {
      return Filters.all();
    }
    return switch (operand) {
      case Group g -> convert(g.content());
      case Expression expr -> convertExpression(expr);
      default ->
          throw new UnsupportedOperationException(
              "Unsupported filter operand type: " + operand.getClass().getName());
    };
  }

  private static Filter convertExpression(Expression expr) {
    return switch (expr.type()) {
      case EQ -> convertEq(expr);
      case NE -> Filters.not(convertEq(expr));
      case GT, GTE, LT, LTE -> convertComparison(expr);
      case IN -> convertIn(expr);
      case NIN -> Filters.not(convertIn(expr));
      case AND -> Filters.and(convert(expr.left()), convert(expr.right()));
      case OR -> Filters.or(convert(expr.left()), convert(expr.right()));
      case NOT -> Filters.not(convert(expr.left()));
      default ->
          throw new UnsupportedOperationException(
              "Unsupported filter expression type: " + expr.type());
    };
  }

  private static Filter convertEq(Expression expr) {
    String field = ((Key) expr.left()).key();
    Object value = ((Value) expr.right()).value();
    return switch (value) {
      case String s -> Filters.eq(field, s);
      case Long l -> Filters.eq(field, l);
      case Integer i -> Filters.eq(field, (long) i);
      case Double d -> Filters.eq(field, d);
      case Float f -> Filters.eq(field, (double) f);
      case Boolean b -> Filters.eq(field, b);
      default -> Filters.eq(field, value.toString());
    };
  }

  private static Filter convertComparison(Expression expr) {
    String field = ((Key) expr.left()).key();
    double value = toDouble(((Value) expr.right()).value());
    return switch (expr.type()) {
      case GT -> Filters.gt(field, value);
      case GTE -> Filters.gte(field, value);
      case LT -> Filters.lt(field, value);
      case LTE -> Filters.lte(field, value);
      default -> throw new IllegalStateException("Not a comparison: " + expr.type());
    };
  }

  private static Filter convertIn(Expression expr) {
    String field = ((Key) expr.left()).key();
    Object rawValues = ((Value) expr.right()).value();
    if (!(rawValues instanceof List<?> list) || list.isEmpty()) {
      throw new IllegalArgumentException("IN filter requires a non-empty list of values");
    }
    Object first = list.getFirst();
    if (first instanceof String) {
      return Filters.inStr(field, list.stream().map(Object::toString).toArray(String[]::new));
    }
    if (first instanceof Number) {
      double[] nums = new double[list.size()];
      for (int i = 0; i < list.size(); i++) {
        nums[i] = ((Number) list.get(i)).doubleValue();
      }
      return Filters.inNum(field, nums);
    }
    // Fallback for mixed or unknown element types
    return new Filter.In(field, list.stream().map(v -> (Object) v).toList());
  }

  private static double toDouble(Object value) {
    return switch (value) {
      case Number n -> n.doubleValue();
      default ->
          throw new IllegalArgumentException(
              "Numeric comparison requires a Number, got: " + value.getClass().getName());
    };
  }
}
