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
package com.integrallis.vectors.spring.ai;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.filter.Filter;
import java.util.List;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.vectorstore.filter.Filter.Expression;
import org.springframework.ai.vectorstore.filter.Filter.ExpressionType;
import org.springframework.ai.vectorstore.filter.Filter.Group;
import org.springframework.ai.vectorstore.filter.Filter.Key;
import org.springframework.ai.vectorstore.filter.Filter.Value;

class FilterConverterTest {

  @Nested
  @Tag("unit")
  class NullAndEmpty {

    @Test
    void nullExpressionReturnsAll() {
      Filter result = FilterConverter.convert(null);
      assertThat(result).isInstanceOf(Filter.All.class);
    }
  }

  @Nested
  @Tag("unit")
  class Equality {

    @Test
    void eqString() {
      Expression expr = new Expression(ExpressionType.EQ, new Key("color"), new Value("red"));
      Filter result = FilterConverter.convert(expr);

      assertThat(result).isInstanceOf(Filter.Eq.class);
      Filter.Eq eq = (Filter.Eq) result;
      assertThat(eq.field()).isEqualTo("color");
      assertThat(eq.value()).isEqualTo("red");
    }

    @Test
    void eqLong() {
      Expression expr = new Expression(ExpressionType.EQ, new Key("count"), new Value(42L));
      Filter result = FilterConverter.convert(expr);

      assertThat(result).isInstanceOf(Filter.Eq.class);
      Filter.Eq eq = (Filter.Eq) result;
      assertThat(eq.field()).isEqualTo("count");
      assertThat(eq.value()).isEqualTo(42L);
    }

    @Test
    void eqDouble() {
      Expression expr = new Expression(ExpressionType.EQ, new Key("score"), new Value(3.14));
      Filter result = FilterConverter.convert(expr);

      assertThat(result).isInstanceOf(Filter.Eq.class);
      Filter.Eq eq = (Filter.Eq) result;
      assertThat(eq.value()).isEqualTo(3.14);
    }

    @Test
    void eqBoolean() {
      Expression expr = new Expression(ExpressionType.EQ, new Key("active"), new Value(true));
      Filter result = FilterConverter.convert(expr);

      assertThat(result).isInstanceOf(Filter.Eq.class);
      Filter.Eq eq = (Filter.Eq) result;
      assertThat(eq.value()).isEqualTo(true);
    }

    @Test
    void neWrapsInNot() {
      Expression expr = new Expression(ExpressionType.NE, new Key("color"), new Value("red"));
      Filter result = FilterConverter.convert(expr);

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
    void gt() {
      Expression expr = new Expression(ExpressionType.GT, new Key("price"), new Value(100.0));
      Filter result = FilterConverter.convert(expr);

      assertThat(result).isInstanceOf(Filter.NumericRange.class);
      Filter.NumericRange nr = (Filter.NumericRange) result;
      assertThat(nr.field()).isEqualTo("price");
      assertThat(nr.lower()).isEqualTo(100.0);
      assertThat(nr.lowerInclusive()).isFalse();
      assertThat(nr.upper()).isNull();
    }

    @Test
    void gte() {
      Expression expr = new Expression(ExpressionType.GTE, new Key("price"), new Value(100.0));
      Filter result = FilterConverter.convert(expr);

      Filter.NumericRange nr = (Filter.NumericRange) result;
      assertThat(nr.lower()).isEqualTo(100.0);
      assertThat(nr.lowerInclusive()).isTrue();
    }

    @Test
    void lt() {
      Expression expr = new Expression(ExpressionType.LT, new Key("price"), new Value(50.0));
      Filter result = FilterConverter.convert(expr);

      Filter.NumericRange nr = (Filter.NumericRange) result;
      assertThat(nr.upper()).isEqualTo(50.0);
      assertThat(nr.upperInclusive()).isFalse();
      assertThat(nr.lower()).isNull();
    }

    @Test
    void lte() {
      Expression expr = new Expression(ExpressionType.LTE, new Key("price"), new Value(50.0));
      Filter result = FilterConverter.convert(expr);

      Filter.NumericRange nr = (Filter.NumericRange) result;
      assertThat(nr.upper()).isEqualTo(50.0);
      assertThat(nr.upperInclusive()).isTrue();
    }

    @Test
    void gtWithInteger() {
      Expression expr = new Expression(ExpressionType.GT, new Key("count"), new Value(10));
      Filter result = FilterConverter.convert(expr);

      Filter.NumericRange nr = (Filter.NumericRange) result;
      assertThat(nr.field()).isEqualTo("count");
      assertThat(nr.lower()).isEqualTo(10.0);
      assertThat(nr.lowerInclusive()).isFalse();
    }
  }

  @Nested
  @Tag("unit")
  class Membership {

    @Test
    void inStrings() {
      Expression expr =
          new Expression(
              ExpressionType.IN, new Key("color"), new Value(List.of("red", "blue", "green")));
      Filter result = FilterConverter.convert(expr);

      assertThat(result).isInstanceOf(Filter.In.class);
      Filter.In in = (Filter.In) result;
      assertThat(in.field()).isEqualTo("color");
      assertThat(in.values()).containsExactly("red", "blue", "green");
    }

    @Test
    void inNumbers() {
      Expression expr =
          new Expression(ExpressionType.IN, new Key("id"), new Value(List.of(1, 2, 3)));
      Filter result = FilterConverter.convert(expr);

      assertThat(result).isInstanceOf(Filter.In.class);
    }

    @Test
    void ninWrapsInNot() {
      Expression expr =
          new Expression(ExpressionType.NIN, new Key("color"), new Value(List.of("red", "blue")));
      Filter result = FilterConverter.convert(expr);

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
      Expression left = new Expression(ExpressionType.EQ, new Key("a"), new Value("x"));
      Expression right = new Expression(ExpressionType.EQ, new Key("b"), new Value("y"));
      Expression expr = new Expression(ExpressionType.AND, left, right);

      Filter result = FilterConverter.convert(expr);

      assertThat(result).isInstanceOf(Filter.And.class);
      Filter.And and = (Filter.And) result;
      assertThat(and.children()).hasSize(2);
      assertThat(and.children().get(0)).isInstanceOf(Filter.Eq.class);
      assertThat(and.children().get(1)).isInstanceOf(Filter.Eq.class);
    }

    @Test
    void or() {
      Expression left = new Expression(ExpressionType.EQ, new Key("a"), new Value("x"));
      Expression right = new Expression(ExpressionType.EQ, new Key("b"), new Value("y"));
      Expression expr = new Expression(ExpressionType.OR, left, right);

      Filter result = FilterConverter.convert(expr);

      assertThat(result).isInstanceOf(Filter.Or.class);
      Filter.Or or = (Filter.Or) result;
      assertThat(or.children()).hasSize(2);
    }

    @Test
    void not() {
      Expression inner = new Expression(ExpressionType.EQ, new Key("a"), new Value("x"));
      Expression expr = new Expression(ExpressionType.NOT, inner);

      Filter result = FilterConverter.convert(expr);

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
      Expression eqColor = new Expression(ExpressionType.EQ, new Key("color"), new Value("red"));
      Expression gtPrice = new Expression(ExpressionType.GT, new Key("price"), new Value(100.0));
      Expression ltPrice = new Expression(ExpressionType.LT, new Key("price"), new Value(10.0));
      Expression orPrices = new Expression(ExpressionType.OR, gtPrice, ltPrice);
      Expression andExpr = new Expression(ExpressionType.AND, eqColor, orPrices);

      Filter result = FilterConverter.convert(andExpr);

      assertThat(result).isInstanceOf(Filter.And.class);
      Filter.And and = (Filter.And) result;
      assertThat(and.children().get(0)).isInstanceOf(Filter.Eq.class);
      assertThat(and.children().get(1)).isInstanceOf(Filter.Or.class);

      Filter.Or or = (Filter.Or) and.children().get(1);
      assertThat(or.children().get(0)).isInstanceOf(Filter.NumericRange.class);
      assertThat(or.children().get(1)).isInstanceOf(Filter.NumericRange.class);
    }

    @Test
    void groupUnwrapping() {
      Expression inner = new Expression(ExpressionType.EQ, new Key("k"), new Value("v"));
      Group group = new Group(inner);
      // Group wraps an expression; convert should unwrap transparently
      Filter result = FilterConverter.convert(group);

      assertThat(result).isInstanceOf(Filter.Eq.class);
    }
  }
}
