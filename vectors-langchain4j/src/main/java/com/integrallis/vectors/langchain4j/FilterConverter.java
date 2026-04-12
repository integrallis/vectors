package com.integrallis.vectors.langchain4j;

import com.integrallis.vectors.db.filter.Filter;
import com.integrallis.vectors.db.filter.Filters;
import dev.langchain4j.store.embedding.filter.comparison.IsEqualTo;
import dev.langchain4j.store.embedding.filter.comparison.IsGreaterThan;
import dev.langchain4j.store.embedding.filter.comparison.IsGreaterThanOrEqualTo;
import dev.langchain4j.store.embedding.filter.comparison.IsIn;
import dev.langchain4j.store.embedding.filter.comparison.IsLessThan;
import dev.langchain4j.store.embedding.filter.comparison.IsLessThanOrEqualTo;
import dev.langchain4j.store.embedding.filter.comparison.IsNotEqualTo;
import dev.langchain4j.store.embedding.filter.comparison.IsNotIn;
import dev.langchain4j.store.embedding.filter.logical.And;
import dev.langchain4j.store.embedding.filter.logical.Not;
import dev.langchain4j.store.embedding.filter.logical.Or;

/**
 * Converts LangChain4j {@link dev.langchain4j.store.embedding.filter.Filter} instances to
 * java-vectors {@link Filter} AST.
 *
 * <p>This class is stateless; all methods are static.
 */
final class FilterConverter {

  private FilterConverter() {}

  /**
   * Converts a LangChain4j filter to a java-vectors filter.
   *
   * @param filter the LangChain4j filter, may be null
   * @return the java-vectors filter, never null ({@link Filter.All} for null input)
   * @throws UnsupportedOperationException for unsupported filter types (e.g. ContainsString)
   */
  static Filter convert(dev.langchain4j.store.embedding.filter.Filter filter) {
    if (filter == null) {
      return Filters.all();
    }
    return switch (filter) {
      case IsEqualTo f -> convertEq(f.key(), f.comparisonValue());
      case IsNotEqualTo f -> Filters.not(convertEq(f.key(), f.comparisonValue()));
      case IsGreaterThan f -> Filters.gt(f.key(), toDouble(f.comparisonValue()));
      case IsGreaterThanOrEqualTo f -> Filters.gte(f.key(), toDouble(f.comparisonValue()));
      case IsLessThan f -> Filters.lt(f.key(), toDouble(f.comparisonValue()));
      case IsLessThanOrEqualTo f -> Filters.lte(f.key(), toDouble(f.comparisonValue()));
      case IsIn f -> convertIn(f.key(), f.comparisonValues());
      case IsNotIn f -> Filters.not(convertIn(f.key(), f.comparisonValues()));
      case And f -> Filters.and(convert(f.left()), convert(f.right()));
      case Or f -> Filters.or(convert(f.left()), convert(f.right()));
      case Not f -> Filters.not(convert(f.expression()));
      default ->
          throw new UnsupportedOperationException(
              "Unsupported LangChain4j filter type: " + filter.getClass().getSimpleName());
    };
  }

  private static Filter convertEq(String field, Object value) {
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

  private static Filter convertIn(String field, java.util.Collection<?> values) {
    Object first = values.iterator().next();
    if (first instanceof String) {
      return Filters.inStr(field, values.stream().map(Object::toString).toArray(String[]::new));
    }
    if (first instanceof Number) {
      double[] nums = new double[values.size()];
      int i = 0;
      for (Object v : values) {
        nums[i++] = ((Number) v).doubleValue();
      }
      return Filters.inNum(field, nums);
    }
    // Fallback for mixed or unknown element types
    return new Filter.In(field, values.stream().map(v -> (Object) v).toList());
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
