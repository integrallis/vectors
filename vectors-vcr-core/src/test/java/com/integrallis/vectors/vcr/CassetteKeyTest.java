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
