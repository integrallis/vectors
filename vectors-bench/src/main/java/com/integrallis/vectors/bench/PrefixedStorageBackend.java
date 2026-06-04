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
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * {@link StorageBackend} decorator that namespaces every key under a fixed {@code root} prefix, so
 * a benchmark run is isolated within a shared bucket and its objects can be listed/deleted as a
 * group. {@link #list} strips the prefix back off so callers see logical keys; {@link #delete}
 * re-applies it (P1.8).
 */
final class PrefixedStorageBackend implements StorageBackend {

  private final StorageBackend delegate;
  private final String root;

  PrefixedStorageBackend(StorageBackend delegate, String root) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
    this.root = Objects.requireNonNull(root, "root");
  }

  String root() {
    return root;
  }

  @Override
  public void put(String key, byte[] value) throws IOException {
    delegate.put(root + key, value);
  }

  @Override
  public byte[] get(String key) throws IOException {
    return delegate.get(root + key);
  }

  @Override
  public StoredValue getWithEtag(String key) throws IOException {
    return delegate.getWithEtag(root + key);
  }

  @Override
  public byte[] getRange(String key, long offset, int length) throws IOException {
    return delegate.getRange(root + key, offset, length);
  }

  @Override
  public List<String> list(String prefix) throws IOException {
    List<String> raw = delegate.list(root + prefix);
    List<String> out = new ArrayList<>(raw.size());
    for (String k : raw) {
      out.add(k.startsWith(root) ? k.substring(root.length()) : k);
    }
    return out;
  }

  @Override
  public void delete(String key) throws IOException {
    delegate.delete(root + key);
  }

  @Override
  public ConditionalPutResult conditionalPut(String key, byte[] value, String expectedEtag)
      throws IOException {
    return delegate.conditionalPut(root + key, value, expectedEtag);
  }
}
