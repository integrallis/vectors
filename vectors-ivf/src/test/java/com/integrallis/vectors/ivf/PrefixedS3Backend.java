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
package com.integrallis.vectors.ivf;

import com.integrallis.vectors.storage.backend.S3StorageBackend;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import software.amazon.awssdk.services.s3.S3Client;

/**
 * Test-only {@link StorageBackend} that namespaces all keys under a fixed prefix on top of a shared
 * bucket. Used by {@link DistributedVectorCollectionR2IT} so multiple invocations against the same
 * Cloudflare R2 bucket — and any unrelated objects the user has placed in that bucket — never
 * collide with each other.
 *
 * <p>Every call rewrites the user-supplied key as {@code prefix + key} on the way down, and strips
 * {@code prefix} from any returned keys on the way back up.
 */
final class PrefixedS3Backend implements StorageBackend {

  private final S3StorageBackend delegate;
  private final String prefix;

  PrefixedS3Backend(S3Client s3, String bucket, String prefix) {
    this.delegate = new S3StorageBackend(s3, bucket);
    this.prefix = prefix;
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
}
