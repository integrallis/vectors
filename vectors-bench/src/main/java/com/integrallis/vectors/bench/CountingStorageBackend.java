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
package com.integrallis.vectors.bench;

import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * {@link StorageBackend} decorator that tallies operation counts and bytes transferred, so the
 * benchmark can report bytes-fetched-per-query and GETs-per-query (P1.8). Counters are resettable
 * per phase/per query via {@link #reset()}. GET/PUT counts map to R2 Class-B/Class-A operations for
 * the cost model.
 */
final class CountingStorageBackend implements StorageBackend {

  private final StorageBackend delegate;
  private final AtomicLong getCalls = new AtomicLong();
  private final AtomicLong putCalls = new AtomicLong();
  private final AtomicLong listCalls = new AtomicLong();
  private final AtomicLong deleteCalls = new AtomicLong();
  private final AtomicLong bytesFetched = new AtomicLong();
  private final AtomicLong bytesPut = new AtomicLong();

  CountingStorageBackend(StorageBackend delegate) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
  }

  /** Zeroes every counter (call before timing a phase or a single query). */
  void reset() {
    getCalls.set(0);
    putCalls.set(0);
    listCalls.set(0);
    deleteCalls.set(0);
    bytesFetched.set(0);
    bytesPut.set(0);
  }

  long getCalls() {
    return getCalls.get();
  }

  long putCalls() {
    return putCalls.get();
  }

  long listCalls() {
    return listCalls.get();
  }

  long deleteCalls() {
    return deleteCalls.get();
  }

  long bytesFetched() {
    return bytesFetched.get();
  }

  long bytesPut() {
    return bytesPut.get();
  }

  @Override
  public void put(String key, byte[] value) throws IOException {
    putCalls.incrementAndGet();
    bytesPut.addAndGet(value.length);
    delegate.put(key, value);
  }

  @Override
  public byte[] get(String key) throws IOException {
    getCalls.incrementAndGet();
    byte[] v = delegate.get(key);
    if (v != null) {
      bytesFetched.addAndGet(v.length);
    }
    return v;
  }

  @Override
  public StoredValue getWithEtag(String key) throws IOException {
    getCalls.incrementAndGet();
    StoredValue v = delegate.getWithEtag(key);
    if (v != null) {
      bytesFetched.addAndGet(v.value().length);
    }
    return v;
  }

  @Override
  public byte[] getRange(String key, long offset, int length) throws IOException {
    getCalls.incrementAndGet();
    byte[] v = delegate.getRange(key, offset, length);
    if (v != null) {
      bytesFetched.addAndGet(v.length);
    }
    return v;
  }

  @Override
  public List<String> list(String prefix) throws IOException {
    listCalls.incrementAndGet();
    return delegate.list(prefix);
  }

  @Override
  public void delete(String key) throws IOException {
    deleteCalls.incrementAndGet();
    delegate.delete(key);
  }

  @Override
  public ConditionalPutResult conditionalPut(String key, byte[] value, String expectedEtag)
      throws IOException {
    putCalls.incrementAndGet();
    bytesPut.addAndGet(value.length);
    return delegate.conditionalPut(key, value, expectedEtag);
  }
}
