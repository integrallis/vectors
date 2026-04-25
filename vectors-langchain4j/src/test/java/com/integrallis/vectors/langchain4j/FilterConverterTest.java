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
package com.integrallis.vectors.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.filter.Filter;
import dev.langchain4j.store.embedding.filter.comparison.ContainsString;
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
import java.util.List;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class FilterConverterTest {

  @Nested
  @Tag("unit")
  class NullFilter {

    @Test
    void nullReturnsAll() {
      Filter result = FilterConverter.convert(null);
      assertThat(result).isInstanceOf(Filter.All.class);
    }
  }

  @Nested
  @Tag("unit")
  class Equality {

    @Test
    void isEqualToString() {
      var lc = new IsEqualTo("color", "red");
      Filter result = FilterConverter.convert(lc);

      assertThat(result).isInstanceOf(Filter.Eq.class);
      Filter.Eq eq = (Filter.Eq) result;
      assertThat(eq.field()).isEqualTo("color");
      assertThat(eq.value()).isEqualTo("red");
    }

    @Test
    void isEqualToLong() {
      var lc = new IsEqualTo("count", 42L);
      Filter result = FilterConverter.convert(lc);

      Filter.Eq eq = (Filter.Eq) result;
      assertThat(eq.field()).isEqualTo("count");
      assertThat(eq.value()).isEqualTo(42L);
    }

    @Test
    void isEqualToDouble() {
      var lc = new IsEqualTo("score", 3.14);
      Filter result = FilterConverter.convert(lc);

      Filter.Eq eq = (Filter.Eq) result;
      assertThat(eq.value()).isEqualTo(3.14);
    }

    @Test
    void isNotEqualToWrapsInNot() {
      var lc = new IsNotEqualTo("color", "red");
      Filter result = FilterConverter.convert(lc);

      assertThat(result).isInstanceOf(Filter.Not.class);
      Filter.Not not = (Filter.Not) result;
      assertThat(not.child()).isInstanceOf(Filter.Eq.class);
      Filter.Eq eq = (Filter.Eq) not.child();
      assertThat(eq.field()).isEqualTo("color");
      assertThat(eq.value()).isEqualTo("red");
    }
  }

  @Nested
  @Tag("unit")
  class NumericComparisons {

    @Test
    void greaterThan() {
      var lc = new IsGreaterThan("price", 100.0);
      Filter result = FilterConverter.convert(lc);

      assertThat(result).isInstanceOf(Filter.NumericRange.class);
      Filter.NumericRange nr = (Filter.NumericRange) result;
      assertThat(nr.field()).isEqualTo("price");
      assertThat(nr.lower()).isEqualTo(100.0);
      assertThat(nr.lowerInclusive()).isFalse();
      assertThat(nr.upper()).isNull();
    }

    @Test
    void greaterThanOrEqualTo() {
      var lc = new IsGreaterThanOrEqualTo("price", 100.0);
      Filter result = FilterConverter.convert(lc);

      Filter.NumericRange nr = (Filter.NumericRange) result;
      assertThat(nr.lower()).isEqualTo(100.0);
      assertThat(nr.lowerInclusive()).isTrue();
    }

    @Test
    void lessThan() {
      var lc = new IsLessThan("price", 50.0);
      Filter result = FilterConverter.convert(lc);

      Filter.NumericRange nr = (Filter.NumericRange) result;
      assertThat(nr.upper()).isEqualTo(50.0);
      assertThat(nr.upperInclusive()).isFalse();
      assertThat(nr.lower()).isNull();
    }

    @Test
    void lessThanOrEqualTo() {
      var lc = new IsLessThanOrEqualTo("price", 50.0);
      Filter result = FilterConverter.convert(lc);

      Filter.NumericRange nr = (Filter.NumericRange) result;
      assertThat(nr.upper()).isEqualTo(50.0);
      assertThat(nr.upperInclusive()).isTrue();
    }

    @Test
    void greaterThanWithInteger() {
      var lc = new IsGreaterThan("count", 10);
      Filter result = FilterConverter.convert(lc);

      Filter.NumericRange nr = (Filter.NumericRange) result;
      assertThat(nr.lower()).isEqualTo(10.0);
    }
  }

  @Nested
  @Tag("unit")
  class Membership {

    @Test
    void isIn() {
      var lc = new IsIn("color", List.of("red", "blue", "green"));
      Filter result = FilterConverter.convert(lc);

      assertThat(result).isInstanceOf(Filter.In.class);
      Filter.In in = (Filter.In) result;
      assertThat(in.field()).isEqualTo("color");
      assertThat(in.values()).containsExactlyInAnyOrder("red", "blue", "green");
    }

    @Test
    void isNotInWrapsInNot() {
      var lc = new IsNotIn("color", List.of("red", "blue"));
      Filter result = FilterConverter.convert(lc);

      assertThat(result).isInstanceOf(Filter.Not.class);
      Filter.Not not = (Filter.Not) result;
      assertThat(not.child()).isInstanceOf(Filter.In.class);
    }
  }

  @Nested
  @Tag("unit")
  class LogicalOperators {

    @Test
    void and() {
      var left = new IsEqualTo("a", "x");
      var right = new IsEqualTo("b", "y");
      var lc = new And(left, right);

      Filter result = FilterConverter.convert(lc);

      assertThat(result).isInstanceOf(Filter.And.class);
      Filter.And and = (Filter.And) result;
      assertThat(and.children()).hasSize(2);
      assertThat(and.children().get(0)).isInstanceOf(Filter.Eq.class);
      assertThat(and.children().get(1)).isInstanceOf(Filter.Eq.class);
    }

    @Test
    void or() {
      var left = new IsEqualTo("a", "x");
      var right = new IsEqualTo("b", "y");
      var lc = new Or(left, right);

      Filter result = FilterConverter.convert(lc);

      assertThat(result).isInstanceOf(Filter.Or.class);
      Filter.Or or = (Filter.Or) result;
      assertThat(or.children()).hasSize(2);
    }

    @Test
    void not() {
      var inner = new IsEqualTo("a", "x");
      var lc = new Not(inner);

      Filter result = FilterConverter.convert(lc);

      assertThat(result).isInstanceOf(Filter.Not.class);
      Filter.Not not = (Filter.Not) result;
      assertThat(not.child()).isInstanceOf(Filter.Eq.class);
    }
  }

  @Nested
  @Tag("unit")
  class Compound {

    @Test
    void nestedAndOrGt() {
      // AND(EQ("color","red"), OR(GT("price",100), LT("price",10)))
      var eqColor = new IsEqualTo("color", "red");
      var gtPrice = new IsGreaterThan("price", 100.0);
      var ltPrice = new IsLessThan("price", 10.0);
      var orPrices = new Or(gtPrice, ltPrice);
      var andExpr = new And(eqColor, orPrices);

      Filter result = FilterConverter.convert(andExpr);

      assertThat(result).isInstanceOf(Filter.And.class);
      Filter.And and = (Filter.And) result;
      assertThat(and.children().get(0)).isInstanceOf(Filter.Eq.class);
      assertThat(and.children().get(1)).isInstanceOf(Filter.Or.class);

      Filter.Or or = (Filter.Or) and.children().get(1);
      assertThat(or.children().get(0)).isInstanceOf(Filter.NumericRange.class);
      assertThat(or.children().get(1)).isInstanceOf(Filter.NumericRange.class);
    }
  }

  @Nested
  @Tag("unit")
  class Unsupported {

    @Test
    void containsStringThrows() {
      var lc = new ContainsString("name", "test");
      assertThatThrownBy(() -> FilterConverter.convert(lc))
          .isInstanceOf(UnsupportedOperationException.class)
          .hasMessageContaining("ContainsString");
    }
  }
}
