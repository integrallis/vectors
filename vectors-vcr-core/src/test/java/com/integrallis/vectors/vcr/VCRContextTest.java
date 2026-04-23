package com.integrallis.vectors.vcr;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class VCRContextTest {

  private VCRContext newContext(VCRMode mode) {
    HeapStorageBackend backend = new HeapStorageBackend();
    ExactCassetteStore store = new ExactCassetteStore(backend, new TestSerializer());
    VCRRegistry registry = new VCRRegistry(backend);
    return new VCRContext(store, registry, mode);
  }

  @Test
  void generateCassetteKeyIncrementsPerType() {
    VCRContext ctx = newContext(VCRMode.RECORD);
    ctx.setCurrentTest("Suite:test1");

    CassetteKey e1 = ctx.generateCassetteKey("embedding");
    CassetteKey e2 = ctx.generateCassetteKey("embedding");
    CassetteKey c1 = ctx.generateCassetteKey("chat");

    assertEquals(1, e1.callIndex());
    assertEquals(2, e2.callIndex());
    assertEquals(1, c1.callIndex());
    assertEquals(3, ctx.getCurrentCassetteKeys().size());
  }

  @Test
  void resetCallCountersClearsState() {
    VCRContext ctx = newContext(VCRMode.RECORD);
    ctx.setCurrentTest("Suite:test1");
    ctx.generateCassetteKey("embedding");
    ctx.resetCallCounters();
    assertTrue(ctx.getCurrentCassetteKeys().isEmpty());
    assertEquals(1, ctx.generateCassetteKey("embedding").callIndex());
  }

  @Test
  void generateCassetteKeyRequiresCurrentTest() {
    VCRContext ctx = newContext(VCRMode.RECORD);
    assertThrows(IllegalStateException.class, () -> ctx.generateCassetteKey("embedding"));
  }

  @Test
  void statisticsAreTracked() {
    VCRContext ctx = newContext(VCRMode.PLAYBACK);
    ctx.recordCacheHit();
    ctx.recordCacheHit();
    ctx.recordCacheMiss();
    ctx.recordApiCall();
    assertEquals(2, ctx.getCacheHits());
    assertEquals(1, ctx.getCacheMisses());
    assertEquals(1, ctx.getApiCalls());
  }

  @Test
  void exposesStoreAndRegistry() {
    VCRContext ctx = newContext(VCRMode.PLAYBACK);
    assertNotNull(ctx.getCassetteStore());
    assertNotNull(ctx.getRegistry());
    assertEquals(VCRMode.PLAYBACK, ctx.getEffectiveMode());
  }

  @Test
  void setEffectiveModeOverrides() {
    VCRContext ctx = newContext(VCRMode.PLAYBACK);
    ctx.setEffectiveMode(VCRMode.RECORD);
    assertEquals(VCRMode.RECORD, ctx.getEffectiveMode());
  }
}
