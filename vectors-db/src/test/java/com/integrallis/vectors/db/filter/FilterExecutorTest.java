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
package com.integrallis.vectors.db.filter;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.db.MetadataValue;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class FilterExecutorTest {

  @Nested
  @Tag("unit")
  class AllFilter {

    @Test
    void matchesEverything() {
      assertThat(FilterExecutor.matches(Filters.all(), Map.of())).isTrue();
      assertThat(FilterExecutor.matches(Filters.all(), Map.of("k", MetadataValue.of("v"))))
          .isTrue();
    }
  }

  @Nested
  @Tag("unit")
  class EqualityFilter {

    @Test
    void stringEquality() {
      Map<String, MetadataValue> md = Map.of("color", MetadataValue.of("red"));
      assertThat(FilterExecutor.matches(Filters.eq("color", "red"), md)).isTrue();
      assertThat(FilterExecutor.matches(Filters.eq("color", "blue"), md)).isFalse();
    }

    @Test
    void numericEqualityLong() {
      Map<String, MetadataValue> md = Map.of("count", MetadataValue.of(42L));
      assertThat(FilterExecutor.matches(Filters.eq("count", 42L), md)).isTrue();
      assertThat(FilterExecutor.matches(Filters.eq("count", 43L), md)).isFalse();
    }

    @Test
    void numericEqualityDouble() {
      Map<String, MetadataValue> md = Map.of("score", MetadataValue.of(3.14));
      assertThat(FilterExecutor.matches(Filters.eq("score", 3.14), md)).isTrue();
      assertThat(FilterExecutor.matches(Filters.eq("score", 2.71), md)).isFalse();
    }

    @Test
    void numericCrossTypeEquality() {
      // long value in Eq vs Num stored as double — 42L and 42.0 should match
      Map<String, MetadataValue> md = Map.of("x", MetadataValue.of(42L));
      assertThat(FilterExecutor.matches(Filters.eq("x", 42L), md)).isTrue();
    }

    @Test
    void booleanEquality() {
      Map<String, MetadataValue> md = Map.of("active", MetadataValue.of(true));
      assertThat(FilterExecutor.matches(Filters.eq("active", true), md)).isTrue();
      assertThat(FilterExecutor.matches(Filters.eq("active", false), md)).isFalse();
    }

    @Test
    void tagsEqMatchesSingleTag() {
      Map<String, MetadataValue> md = Map.of("tags", MetadataValue.tags("a", "b", "c"));
      // Eq on Tags matches if the value is contained in the tag list
      assertThat(FilterExecutor.matches(Filters.eq("tags", "b"), md)).isTrue();
      assertThat(FilterExecutor.matches(Filters.eq("tags", "d"), md)).isFalse();
    }

    @Test
    void missingFieldReturnsFalse() {
      assertThat(FilterExecutor.matches(Filters.eq("missing", "x"), Map.of())).isFalse();
    }

    @Test
    void typeMismatchReturnsFalse() {
      Map<String, MetadataValue> md = Map.of("name", MetadataValue.of("hello"));
      // String metadata vs numeric Eq
      assertThat(FilterExecutor.matches(Filters.eq("name", 42L), md)).isFalse();
    }
  }

  @Nested
  @Tag("unit")
  class NumericRangeFilter {

    private final Map<String, MetadataValue> md = Map.of("age", MetadataValue.of(25.0));

    @Test
    void greaterThanOrEqual() {
      assertThat(FilterExecutor.matches(Filters.gte("age", 25.0), md)).isTrue();
      assertThat(FilterExecutor.matches(Filters.gte("age", 24.0), md)).isTrue();
      assertThat(FilterExecutor.matches(Filters.gte("age", 26.0), md)).isFalse();
    }

    @Test
    void lessThanOrEqual() {
      assertThat(FilterExecutor.matches(Filters.lte("age", 25.0), md)).isTrue();
      assertThat(FilterExecutor.matches(Filters.lte("age", 26.0), md)).isTrue();
      assertThat(FilterExecutor.matches(Filters.lte("age", 24.0), md)).isFalse();
    }

    @Test
    void strictGreaterThan() {
      assertThat(FilterExecutor.matches(Filters.gt("age", 24.0), md)).isTrue();
      assertThat(FilterExecutor.matches(Filters.gt("age", 25.0), md)).isFalse();
    }

    @Test
    void strictLessThan() {
      assertThat(FilterExecutor.matches(Filters.lt("age", 26.0), md)).isTrue();
      assertThat(FilterExecutor.matches(Filters.lt("age", 25.0), md)).isFalse();
    }

    @Test
    void inclusiveRange() {
      assertThat(FilterExecutor.matches(Filters.between("age", 20.0, 30.0), md)).isTrue();
      assertThat(FilterExecutor.matches(Filters.between("age", 25.0, 25.0), md)).isTrue();
      assertThat(FilterExecutor.matches(Filters.between("age", 26.0, 30.0), md)).isFalse();
      assertThat(FilterExecutor.matches(Filters.between("age", 10.0, 24.0), md)).isFalse();
    }

    @Test
    void missingFieldReturnsFalse() {
      assertThat(FilterExecutor.matches(Filters.gte("missing", 0.0), Map.of())).isFalse();
    }

    @Test
    void nonNumericFieldReturnsFalse() {
      Map<String, MetadataValue> strMd = Map.of("name", MetadataValue.of("alice"));
      assertThat(FilterExecutor.matches(Filters.gte("name", 0.0), strMd)).isFalse();
    }
  }

  @Nested
  @Tag("unit")
  class InFilter {

    @Test
    void stringIn() {
      Map<String, MetadataValue> md = Map.of("color", MetadataValue.of("red"));
      assertThat(FilterExecutor.matches(Filters.inStr("color", "red", "blue"), md)).isTrue();
      assertThat(FilterExecutor.matches(Filters.inStr("color", "blue", "green"), md)).isFalse();
    }

    @Test
    void numericIn() {
      Map<String, MetadataValue> md = Map.of("code", MetadataValue.of(42.0));
      assertThat(FilterExecutor.matches(Filters.inNum("code", 42.0, 43.0), md)).isTrue();
      assertThat(FilterExecutor.matches(Filters.inNum("code", 43.0, 44.0), md)).isFalse();
    }

    @Test
    void tagsIntersectsInSet() {
      Map<String, MetadataValue> md = Map.of("labels", MetadataValue.tags("a", "b", "c"));
      assertThat(FilterExecutor.matches(Filters.inStr("labels", "b", "d"), md)).isTrue();
      assertThat(FilterExecutor.matches(Filters.inStr("labels", "x", "y"), md)).isFalse();
    }

    @Test
    void missingFieldReturnsFalse() {
      assertThat(FilterExecutor.matches(Filters.inStr("missing", "a"), Map.of())).isFalse();
    }
  }

  @Nested
  @Tag("unit")
  class LogicalOperators {

    private final Map<String, MetadataValue> md =
        Map.of("color", MetadataValue.of("red"), "size", MetadataValue.of(10.0));

    @Test
    void andBothTrue() {
      Filter f = Filters.and(Filters.eq("color", "red"), Filters.gte("size", 5.0));
      assertThat(FilterExecutor.matches(f, md)).isTrue();
    }

    @Test
    void andOneFalse() {
      Filter f = Filters.and(Filters.eq("color", "red"), Filters.gte("size", 20.0));
      assertThat(FilterExecutor.matches(f, md)).isFalse();
    }

    @Test
    void orOneTrue() {
      Filter f = Filters.or(Filters.eq("color", "blue"), Filters.gte("size", 5.0));
      assertThat(FilterExecutor.matches(f, md)).isTrue();
    }

    @Test
    void orNoneTrue() {
      Filter f = Filters.or(Filters.eq("color", "blue"), Filters.gte("size", 20.0));
      assertThat(FilterExecutor.matches(f, md)).isFalse();
    }

    @Test
    void notNegatesChild() {
      assertThat(FilterExecutor.matches(Filters.not(Filters.eq("color", "red")), md)).isFalse();
      assertThat(FilterExecutor.matches(Filters.not(Filters.eq("color", "blue")), md)).isTrue();
    }

    @Test
    void notOnMissingFieldReturnsTrue() {
      // NOT(missing == "x") → NOT(false) → true
      assertThat(FilterExecutor.matches(Filters.not(Filters.eq("absent", "x")), md)).isTrue();
    }

    @Test
    void nestedCompound() {
      // (color == "red" AND size > 5) OR (color == "blue")
      Filter f =
          Filters.or(
              Filters.and(Filters.eq("color", "red"), Filters.gt("size", 5.0)),
              Filters.eq("color", "blue"));
      assertThat(FilterExecutor.matches(f, md)).isTrue();
    }

    @Test
    void deeplyNestedNot() {
      // NOT(NOT(color == "red")) → true
      Filter f = Filters.not(Filters.not(Filters.eq("color", "red")));
      assertThat(FilterExecutor.matches(f, md)).isTrue();
    }
  }

  @Nested
  @Tag("unit")
  class EdgeCases {

    @Test
    void emptyMetadataMatchesAllOnly() {
      Map<String, MetadataValue> empty = Map.of();
      assertThat(FilterExecutor.matches(Filters.all(), empty)).isTrue();
      assertThat(FilterExecutor.matches(Filters.eq("x", "y"), empty)).isFalse();
    }

    @Test
    void nullMetadataMatchesAllOnly() {
      assertThat(FilterExecutor.matches(Filters.all(), null)).isTrue();
      assertThat(FilterExecutor.matches(Filters.eq("x", "y"), null)).isFalse();
      assertThat(FilterExecutor.matches(Filters.not(Filters.eq("x", "y")), null)).isFalse();
    }

    @Test
    void boolInFilter() {
      Map<String, MetadataValue> md = Map.of("flag", MetadataValue.of(true));
      Filter inFilter = new Filter.In("flag", List.of(true, false));
      assertThat(FilterExecutor.matches(inFilter, md)).isTrue();
    }

    @Test
    void numericRangeOnIntegerStoredValue() {
      // MetadataValue.of(42L) stores as Num(42.0) — should match range [40, 45]
      Map<String, MetadataValue> md = Map.of("count", MetadataValue.of(42L));
      assertThat(FilterExecutor.matches(Filters.between("count", 40.0, 45.0), md)).isTrue();
    }

    @Test
    void eqWithDoubleMatchesLongStored() {
      // Filter.Eq with double 42.0 vs Num(42.0) stored from long
      Map<String, MetadataValue> md = Map.of("x", MetadataValue.of(42L));
      assertThat(FilterExecutor.matches(Filters.eq("x", 42.0), md)).isTrue();
    }
  }
}
