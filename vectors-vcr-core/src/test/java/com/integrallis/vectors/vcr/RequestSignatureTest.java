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
package com.integrallis.vectors.vcr;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RequestSignatureTest {

  @Test
  void canonicalRequestTreesProduceStableOrderIndependentSignatures() {
    Map<String, Object> first = new LinkedHashMap<>();
    first.put("text", new StringBuilder("hello"));
    first.put("character", 'x');
    first.put("number", 42);
    first.put("enabled", true);
    first.put("mode", VCRMode.PLAYBACK_OR_RECORD);
    first.put("items", List.of("one", 2L));
    first.put("primitiveArray", new int[] {3, 4});
    first.put("objectArray", new Object[] {"five", null});

    Map<String, Object> second = new LinkedHashMap<>();
    first.entrySet().stream()
        .sorted(Map.Entry.<String, Object>comparingByKey().reversed())
        .forEach(entry -> second.put(entry.getKey(), entry.getValue()));

    String signature = RequestSignature.create(first);

    assertThat(signature).startsWith("sha256:").hasSize(71);
    assertThat(RequestSignature.create(second)).isEqualTo(signature);
  }

  @Test
  void typeValueAndSequenceChangesAlterTheSignature() {
    assertThat(RequestSignature.create(Map.of("value", 1)))
        .isNotEqualTo(RequestSignature.create(Map.of("value", 1L)));
    assertThat(RequestSignature.create(List.of("a", "b")))
        .isNotEqualTo(RequestSignature.create(List.of("b", "a")));
    assertThat(RequestSignature.create(null)).isNotEqualTo(RequestSignature.create("null"));
  }

  @Test
  void unsupportedValuesFailInsteadOfCreatingUnstableSignatures() {
    Object unsupported = new Object();

    assertThatIllegalArgumentException()
        .isThrownBy(() -> RequestSignature.create(unsupported))
        .withMessageContaining(Object.class.getName());
  }
}
