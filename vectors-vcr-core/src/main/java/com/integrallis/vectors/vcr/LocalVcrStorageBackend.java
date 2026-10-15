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

import com.integrallis.vectors.storage.backend.LocalFileStorageBackend;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * Portable local storage for VCR logical keys, including registry entries.
 *
 * <p>Logical {@code vcr:...} keys remain unchanged. Their UTF-16 code units are encoded as
 * lowercase hexadecimal in bounded path components under {@code vcr-files-v1}. This encoding is
 * reversible, preserves case and unusual test identifiers, and avoids platform-reserved filenames.
 * Existing colon-named Unix recordings remain readable; new-format values take precedence. Writes
 * migrate a legacy value only after the portable value has been committed. Remote and custom
 * backends do not need this adapter.
 */
public final class LocalVcrStorageBackend implements StorageBackend {
  private static final String DIRECTORY = "vcr-files-v1/";
  private static final String HEX = "0123456789abcdef";
  private final LocalFileStorageBackend files;

  /** Creates a portable cassette and registry store under the supplied directory. */
  public LocalVcrStorageBackend(Path root) throws IOException {
    files = new LocalFileStorageBackend(root);
  }

  @Override
  public synchronized void put(String key, byte[] value) throws IOException {
    files.put(encoded(key), value);
    deleteLegacy(key);
  }

  @Override
  public byte[] get(String key) throws IOException {
    StoredValue result = getWithEtag(key);
    return result == null ? null : result.value();
  }

  @Override
  public StoredValue getWithEtag(String key) throws IOException {
    StoredValue result = files.getWithEtag(encoded(key));
    if (result != null) return result;
    if (!legacyName(key)) return null;
    return files.getWithEtag(key);
  }

  @Override
  public byte[] getRange(String key, long offset, int length) throws IOException {
    String encoded = encoded(key);
    byte[] result = files.getRange(encoded, offset, length);
    if (result != null) return result;
    if (!legacyName(key)) return null;
    return files.getRange(key, offset, length);
  }

  @Override
  public List<String> list(String prefix) throws IOException {
    var keys = new LinkedHashSet<String>();
    for (String physical : files.list(DIRECTORY)) {
      String key = decoded(physical);
      if (key != null && key.startsWith(prefix)) keys.add(key);
    }
    for (String legacy : files.list("vcr:")) {
      if (legacy.startsWith(prefix) && !legacy.endsWith(".etag")) keys.add(legacy);
    }
    return new ArrayList<>(keys);
  }

  @Override
  public synchronized void delete(String key) throws IOException {
    String encoded = encoded(key);
    files.delete(encoded);
    deleteLegacy(key);
  }

  @Override
  public synchronized ConditionalPutResult conditionalPut(
      String key, byte[] value, String expectedEtag) throws IOException {
    StoredValue current = getWithEtag(key);
    String etag = current == null ? null : current.etag();
    if (!java.util.Objects.equals(etag, expectedEtag)) return new ConditionalPutResult(false, null);
    files.put(encoded(key), value);
    deleteLegacy(key);
    return new ConditionalPutResult(true, files.getWithEtag(encoded(key)).etag());
  }

  private void deleteLegacy(String key) throws IOException {
    if (!legacyName(key)) return;
    files.delete(key);
  }

  private static boolean legacyName(String key) {
    try {
      Path legacy = Path.of(key);
      if (legacy.isAbsolute()) return false;
      for (Path part : legacy) {
        if (part.toString().equals("..")) return false;
      }
      return true;
    } catch (InvalidPathException e) {
      return false;
    }
  }

  private static String encoded(String key) throws IOException {
    if (key == null || !key.startsWith("vcr:")) throw new IOException("Expected a VCR logical key");
    StringBuilder out = new StringBuilder(DIRECTORY);
    int digits = 0;
    for (int index = 0; index < key.length(); index++) {
      if (digits == 128) {
        out.append('/');
        digits = 0;
      }
      char value = key.charAt(index);
      for (int shift = 12; shift >= 0; shift -= 4) out.append(HEX.charAt((value >>> shift) & 15));
      digits += 4;
    }
    return out.append("/value").toString();
  }

  private static String decoded(String physical) throws IOException {
    if (!physical.endsWith("/value")) return null;
    String hex = physical.substring(DIRECTORY.length(), physical.length() - 6).replace("/", "");
    if (hex.length() % 4 != 0) return null;
    StringBuilder key = new StringBuilder();
    for (int index = 0; index < hex.length(); index += 4) {
      int value = 0;
      for (int part = 0; part < 4; part++) {
        int digit = HEX.indexOf(hex.charAt(index + part));
        if (digit < 0) return null;
        value = (value << 4) | digit;
      }
      key.append((char) value);
    }
    String result = key.toString();
    return result.startsWith("vcr:") && encoded(result).equals(physical) ? result : null;
  }
}
