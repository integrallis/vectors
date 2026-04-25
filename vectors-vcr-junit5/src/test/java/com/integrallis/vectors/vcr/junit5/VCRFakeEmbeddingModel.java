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
