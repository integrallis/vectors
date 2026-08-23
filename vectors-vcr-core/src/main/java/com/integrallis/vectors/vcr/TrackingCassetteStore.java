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

import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Context-owned cassette-store decorator that reports successful writes to the test lifecycle.
 *
 * <p>The framework adapters create cassette keys themselves, so observing writes at the store
 * boundary keeps cleanup independent of the model framework and automatically covers future wrapper
 * providers.
 */
class TrackingCassetteStore implements CassetteStore {

  private final CassetteStore delegate;
  private final Consumer<CassetteKey> writeTracker;

  private TrackingCassetteStore(CassetteStore delegate, Consumer<CassetteKey> writeTracker) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
    this.writeTracker = Objects.requireNonNull(writeTracker, "writeTracker");
  }

  static CassetteStore wrap(CassetteStore delegate, Consumer<CassetteKey> writeTracker) {
    Objects.requireNonNull(delegate, "delegate");
    Objects.requireNonNull(writeTracker, "writeTracker");
    if (delegate instanceof SimilarityCassetteStore similarityStore) {
      return new TrackingSimilarityCassetteStore(similarityStore, writeTracker);
    }
    return new TrackingCassetteStore(delegate, writeTracker);
  }

  @Override
  public void store(CassetteKey key, CassetteRecord record) {
    delegate.store(key, record);
    writeTracker.accept(key);
  }

  @Override
  public Optional<CassetteRecord> retrieve(CassetteKey key) {
    return delegate.retrieve(key);
  }

  @Override
  public boolean exists(CassetteKey key) {
    return delegate.exists(key);
  }

  @Override
  public void delete(CassetteKey key) {
    delegate.delete(key);
  }

  @Override
  public List<CassetteKey> listByTestId(String testId) {
    return delegate.listByTestId(testId);
  }

  @Override
  public void flush() throws IOException {
    delegate.flush();
  }

  @Override
  public void close() throws IOException {
    delegate.close();
  }

  private static final class TrackingSimilarityCassetteStore extends TrackingCassetteStore
      implements SimilarityCassetteStore {

    private final SimilarityCassetteStore similarityDelegate;

    private TrackingSimilarityCassetteStore(
        SimilarityCassetteStore delegate, Consumer<CassetteKey> writeTracker) {
      super(delegate, writeTracker);
      this.similarityDelegate = delegate;
    }

    @Override
    public Optional<CassetteRecord> retrieveSimilar(float[] queryEmbedding) {
      return similarityDelegate.retrieveSimilar(queryEmbedding);
    }
  }
}
