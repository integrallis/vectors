package com.integrallis.vectors.vcr.junit5;

import com.integrallis.vectors.vcr.CassetteKey;
import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.VCRCassetteMissingException;
import com.integrallis.vectors.vcr.VCRMode;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * VCR wrapper around a {@link FakeEmbeddingModel}. Delegates to the underlying model in record
 * modes and reads stored cassettes in playback modes. The {@code callIndex} mirrors the counter
 * logic used by the real model wrappers.
 */
final class VCRFakeEmbeddingModel implements FakeEmbeddingModel {

  private final FakeEmbeddingModel delegate;
  private final CassetteStore store;
  private final String testId;
  private final VCRMode mode;
  private final AtomicInteger callIndex = new AtomicInteger();

  VCRFakeEmbeddingModel(
      FakeEmbeddingModel delegate, String testId, VCRMode mode, CassetteStore store) {
    this.delegate = delegate;
    this.store = store;
    this.testId = testId;
    this.mode = mode;
  }

  @Override
  public float[] embed(String prompt) {
    int idx = callIndex.incrementAndGet();
    CassetteKey key = new CassetteKey("embedding", testId, idx);
    if (mode == VCRMode.OFF) {
      return delegate.embed(prompt);
    }
    if (mode == VCRMode.PLAYBACK) {
      Optional<CassetteRecord> hit = store.retrieve(key);
      if (hit.isPresent()) {
        return ((CassetteRecord.Embedding) hit.get()).embedding();
      }
      throw new VCRCassetteMissingException(key.serializedKey(), testId);
    }
    float[] out = delegate.embed(prompt);
    store.store(key, new CassetteRecord.Embedding(testId, "fake", System.currentTimeMillis(), out));
    return out;
  }
}
