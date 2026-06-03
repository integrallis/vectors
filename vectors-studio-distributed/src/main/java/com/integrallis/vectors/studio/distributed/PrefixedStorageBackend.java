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
package com.integrallis.vectors.studio.distributed;

import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * {@link StorageBackend} decorator that namespaces all keys under a fixed prefix on top of any
 * delegate backend. Used by {@link DistributedStudioBackend} so a single Cloudflare R2 bucket can
 * host multiple Studio collections (one per prefix) without their object listings colliding.
 */
public final class PrefixedStorageBackend implements StorageBackend, Closeable {

  private final StorageBackend delegate;
  private final String prefix;

  public PrefixedStorageBackend(StorageBackend delegate, String prefix) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
    String p = Objects.requireNonNull(prefix, "prefix");
    this.prefix = p.isEmpty() || p.endsWith("/") ? p : p + "/";
  }

  @Override
  public void put(String key, byte[] value) throws IOException {
    delegate.put(prefix + key, value);
  }

  @Override
  public byte[] get(String key) throws IOException {
    return delegate.get(prefix + key);
  }

  @Override
  public StoredValue getWithEtag(String key) throws IOException {
    return delegate.getWithEtag(prefix + key);
  }

  @Override
  public byte[] getRange(String key, long offset, int length) throws IOException {
    return delegate.getRange(prefix + key, offset, length);
  }

  @Override
  public List<String> list(String keyPrefix) throws IOException {
    List<String> raw = delegate.list(prefix + keyPrefix);
    List<String> out = new ArrayList<>(raw.size());
    for (String k : raw) {
      out.add(k.startsWith(prefix) ? k.substring(prefix.length()) : k);
    }
    return out;
  }

  @Override
  public void delete(String key) throws IOException {
    delegate.delete(prefix + key);
  }

  @Override
  public ConditionalPutResult conditionalPut(String key, byte[] value, String expectedEtag)
      throws IOException {
    return delegate.conditionalPut(prefix + key, value, expectedEtag);
  }

  @Override
  public void close() throws IOException {
    if (delegate instanceof Closeable c) c.close();
  }
}
