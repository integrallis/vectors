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
package com.integrallis.vectors.cache;

import java.util.List;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Factory methods that produce {@link CacheAdmissionPolicy CacheAdmissionPolicy&lt;String&gt;}
 * instances for filtering LLM responses before caching. The default {@link #rejectRefusals()}
 * policy is a sensible out-of-the-box choice for LLM response caching.
 */
public final class LLMResponseFilters {

  private static final List<String> REFUSAL_PHRASES =
      List.of(
          "unable to",
          "i can't",
          "i cannot",
          "please specify",
          "not available",
          "i don't have",
          "i'm sorry, but i can't",
          "visual content is not available",
          "specify which",
          "which image",
          "which page");

  private LLMResponseFilters() {}

  /**
   * Returns a policy that rejects null, blank, and LLM error/refusal/clarification responses.
   * Recognized refusal phrases (case-insensitive):
   *
   * <ul>
   *   <li>"unable to"
   *   <li>"i can't" / "i cannot"
   *   <li>"please specify" / "specify which"
   *   <li>"not available" / "visual content is not available"
   *   <li>"i don't have"
   *   <li>"i'm sorry, but i can't"
   *   <li>"which image" / "which page"
   * </ul>
   */
  public static CacheAdmissionPolicy<String> rejectRefusals() {
    return value -> {
      if (value == null || value.isBlank()) {
        return false;
      }
      String lower = value.toLowerCase();
      for (String phrase : REFUSAL_PHRASES) {
        if (lower.contains(phrase)) {
          return false;
        }
      }
      return true;
    };
  }

  /**
   * Returns a policy that rejects responses shorter than {@code minLength} characters (after
   * trimming).
   *
   * @param minLength minimum acceptable length (inclusive)
   * @throws IllegalArgumentException if minLength is negative
   */
  public static CacheAdmissionPolicy<String> rejectShort(int minLength) {
    if (minLength < 0) {
      throw new IllegalArgumentException("minLength must be non-negative, got " + minLength);
    }
    return value -> value != null && value.trim().length() >= minLength;
  }

  /**
   * Returns a policy that rejects responses containing any of the given phrases (case-insensitive).
   *
   * @param phrases one or more phrases to reject
   * @throws NullPointerException if phrases is null
   * @throws IllegalArgumentException if no phrases are provided
   */
  public static CacheAdmissionPolicy<String> rejectContaining(String... phrases) {
    Objects.requireNonNull(phrases, "phrases");
    if (phrases.length == 0) {
      throw new IllegalArgumentException("at least one phrase is required");
    }
    // Defensive copy and lowercase
    String[] lower = new String[phrases.length];
    for (int i = 0; i < phrases.length; i++) {
      lower[i] = Objects.requireNonNull(phrases[i], "phrases[" + i + "]").toLowerCase();
    }
    return value -> {
      if (value == null) {
        return false;
      }
      String valueLower = value.toLowerCase();
      for (String phrase : lower) {
        if (valueLower.contains(phrase)) {
          return false;
        }
      }
      return true;
    };
  }

  /**
   * Returns a policy that rejects responses matching the given regex pattern.
   *
   * @param regex regular expression; a match anywhere in the response triggers rejection
   * @throws NullPointerException if regex is null
   */
  public static CacheAdmissionPolicy<String> rejectMatching(String regex) {
    Objects.requireNonNull(regex, "regex");
    Pattern pattern = Pattern.compile(regex);
    return value -> value != null && !pattern.matcher(value).find();
  }
}
