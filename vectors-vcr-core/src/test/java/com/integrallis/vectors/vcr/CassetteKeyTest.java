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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class CassetteKeyTest {

  @Test
  void serializesWithZeroPaddedIndex() {
    CassetteKey key = new CassetteKey("embedding", "MyTest:method", 7);
    assertEquals("vcr:embedding:MyTest:method:0007", key.serializedKey());
  }

  @Test
  void roundTripsThroughParse() {
    CassetteKey original = new CassetteKey("chat", "SuiteX:methodY", 42);
    CassetteKey parsed = CassetteKey.parse(original.serializedKey());
    assertEquals(original, parsed);
  }

  @Test
  void parseReturnsNullForInvalidInput() {
    assertNull(CassetteKey.parse(null));
    assertNull(CassetteKey.parse("wrong:prefix:foo:0001"));
    assertNull(CassetteKey.parse("vcr:embedding:only3"));
    assertNull(CassetteKey.parse("vcr:embedding:test:not_a_number"));
  }
}
