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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

@Tag("unit")
class LLMResponseFiltersTest {

  @ParameterizedTest
  @ValueSource(
      strings = {
        "I'm unable to answer that question.",
        "I can't provide that information.",
        "I cannot do that.",
        "Please specify which document you mean.",
        "That information is not available.",
        "I don't have access to that data.",
        "I'm sorry, but I can't help with that.",
        "Visual content is not available for this document.",
        "Could you specify which section you mean?",
        "Which image are you referring to?",
        "Which page would you like me to look at?"
      })
  void rejectRefusalsRejectsEachKnownPattern(String refusal) {
    CacheAdmissionPolicy<String> policy = LLMResponseFilters.rejectRefusals();
    assertThat(policy.test(refusal)).as("Should reject: %s", refusal).isFalse();
  }

  @Test
  void rejectRefusalsAcceptsNormalResponse() {
    CacheAdmissionPolicy<String> policy = LLMResponseFilters.rejectRefusals();
    assertThat(policy.test("The capital of France is Paris.")).isTrue();
    assertThat(policy.test("Here are the key findings from the document...")).isTrue();
  }

  @Test
  void rejectRefusalsRejectsNullAndBlank() {
    CacheAdmissionPolicy<String> policy = LLMResponseFilters.rejectRefusals();
    assertThat(policy.test(null)).isFalse();
    assertThat(policy.test("")).isFalse();
    assertThat(policy.test("   ")).isFalse();
  }

  @Test
  void rejectRefusalsIsCaseInsensitive() {
    CacheAdmissionPolicy<String> policy = LLMResponseFilters.rejectRefusals();
    assertThat(policy.test("I CAN'T do that")).isFalse();
    assertThat(policy.test("UNABLE TO process")).isFalse();
  }

  @Test
  void rejectShortRejectsTooShort() {
    CacheAdmissionPolicy<String> policy = LLMResponseFilters.rejectShort(20);
    assertThat(policy.test("short")).isFalse();
    assertThat(policy.test("")).isFalse();
    assertThat(policy.test(null)).isFalse();
  }

  @Test
  void rejectShortAcceptsLongEnough() {
    CacheAdmissionPolicy<String> policy = LLMResponseFilters.rejectShort(5);
    assertThat(policy.test("hello")).isTrue();
    assertThat(policy.test("this is a long response")).isTrue();
  }

  @Test
  void rejectShortTrimsBeforeChecking() {
    CacheAdmissionPolicy<String> policy = LLMResponseFilters.rejectShort(5);
    assertThat(policy.test("   hi   ")).isFalse(); // trimmed length is 2
  }

  @Test
  void rejectContainingMatchesCaseInsensitive() {
    CacheAdmissionPolicy<String> policy = LLMResponseFilters.rejectContaining("error", "failed");
    assertThat(policy.test("An ERROR occurred")).isFalse();
    assertThat(policy.test("The operation FAILED")).isFalse();
    assertThat(policy.test("Everything went fine")).isTrue();
  }

  @Test
  void rejectContainingRejectsNull() {
    CacheAdmissionPolicy<String> policy = LLMResponseFilters.rejectContaining("error");
    assertThat(policy.test(null)).isFalse();
  }

  @Test
  void rejectMatchingUsesRegex() {
    CacheAdmissionPolicy<String> policy = LLMResponseFilters.rejectMatching("\\berror\\b");
    assertThat(policy.test("An error occurred")).isFalse();
    assertThat(policy.test("No errors here")).isTrue(); // "errors" != "error" word boundary
    assertThat(policy.test(null)).isFalse();
  }

  @Test
  void compositionChainsMultipleFilters() {
    CacheAdmissionPolicy<String> policy =
        LLMResponseFilters.rejectRefusals().and(LLMResponseFilters.rejectShort(20));

    // Normal response, long enough -> admitted
    assertThat(policy.test("The document covers advanced topics in machine learning.")).isTrue();
    // Refusal -> rejected by first policy
    assertThat(policy.test("I can't answer that question about the document.")).isFalse();
    // Too short -> rejected by second policy
    assertThat(policy.test("Yes.")).isFalse();
  }

  @Test
  void rejectRefusalsDoesNotRejectSubstringsInNormalText() {
    CacheAdmissionPolicy<String> policy = LLMResponseFilters.rejectRefusals();
    // "page" by itself should not trigger "which page" pattern
    assertThat(policy.test("The results are on page 42 of the report.")).isTrue();
    // "image" by itself should not trigger "which image" pattern
    List.of(
            "The image shows a bar chart comparing quarterly revenue.",
            "This image was extracted from page 3.")
        .forEach(text -> assertThat(policy.test(text)).as("Should accept: %s", text).isTrue());
  }
}
