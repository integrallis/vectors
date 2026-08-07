/*
 * Copyright 2026 Integrallis Software, LLC
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
package com.integrallis.vectors.cache;

import java.util.Map;
import java.util.Objects;

/**
 * Scopes a semantic lookup to entries whose attributes match.
 *
 * <p>Similarity alone is not enough to decide that a cached answer may be served. A completion
 * generated at temperature 0.9, for a different tenant, or by a different model, can be an
 * excellent semantic match for the current prompt and still be the wrong thing to return. Entries
 * carry attributes describing the conditions they were produced under, and a filter names the
 * conditions the caller will accept.
 *
 * <p>A filter constrains only the attributes it mentions; anything else on the entry is free to
 * vary.
 */
@FunctionalInterface
public interface CacheFilter {

  /**
   * Tests one entry's attributes.
   *
   * @param attributes attributes stored with the entry, never null
   * @return whether the entry may be served
   */
  boolean test(Map<String, String> attributes);

  /**
   * A filter that accepts every entry.
   *
   * @return an unconstrained filter
   */
  static CacheFilter any() {
    return attributes -> true;
  }

  /**
   * Requires one attribute to hold an exact value.
   *
   * @param name attribute name
   * @param value required value
   * @return a filter matching that single attribute
   */
  static CacheFilter matching(String name, String value) {
    Objects.requireNonNull(name, "name");
    Objects.requireNonNull(value, "value");
    return attributes -> value.equals(attributes.get(name));
  }

  /**
   * Requires every supplied attribute to hold its exact value.
   *
   * @param required attributes that must all match
   * @return a conjunctive filter
   */
  static CacheFilter all(Map<String, String> required) {
    Map<String, String> copy = Map.copyOf(Objects.requireNonNull(required, "required"));
    return attributes -> {
      for (Map.Entry<String, String> entry : copy.entrySet()) {
        if (!entry.getValue().equals(attributes.get(entry.getKey()))) {
          return false;
        }
      }
      return true;
    };
  }

  /**
   * Requires this filter and another to both accept.
   *
   * @param other the additional filter
   * @return the conjunction
   */
  default CacheFilter and(CacheFilter other) {
    Objects.requireNonNull(other, "other");
    return attributes -> test(attributes) && other.test(attributes);
  }
}
