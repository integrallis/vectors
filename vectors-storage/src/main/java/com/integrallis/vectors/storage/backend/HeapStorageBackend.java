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
package com.integrallis.vectors.storage.backend;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * In-heap {@link StorageBackend} backed by a {@link ConcurrentHashMap}. Suitable for unit tests and
 * in-process simulation. Not persistent across JVM restarts.
 *
 * <p>Conditional-put uses a synchronized block on the internal map entry to provide atomic
 * compare-and-swap semantics within a single JVM.
 */
public final class HeapStorageBackend implements StorageBackend {

  private final ConcurrentHashMap<String, Entry> store = new ConcurrentHashMap<>();

  @Override
  public synchronized void put(String key, byte[] value) {
    store.put(key, Entry.copyOf(value));
  }

  @Override
  public byte[] get(String key) {
    Entry entry = store.get(key);
    return entry == null ? null : entry.valueCopy();
  }

  @Override
  public byte[] getRange(String key, long offset, int length) {
    Entry entry = store.get(key);
    if (entry == null) return null;
    if (offset < 0 || length < 0 || offset + length > entry.value.length) {
      throw new IndexOutOfBoundsException(
          "getRange("
              + key
              + ", offset="
              + offset
              + ", length="
              + length
              + ") size="
              + entry.value.length);
    }
    int from = Math.toIntExact(offset);
    return Arrays.copyOfRange(entry.value, from, from + length);
  }

  @Override
  public List<String> list(String prefix) {
    List<String> result = new ArrayList<>();
    for (String key : store.keySet()) {
      if (key.startsWith(prefix)) result.add(key);
    }
    return result;
  }

  @Override
  public synchronized void delete(String key) {
    store.remove(key);
  }

  @Override
  public synchronized ConditionalPutResult conditionalPut(
      String key, byte[] value, String expectedEtag) {
    Entry current = store.get(key);
    String currentEtag = current == null ? null : current.etag;
    if (!etagEquals(currentEtag, expectedEtag)) {
      return new ConditionalPutResult(false, null);
    }
    Entry next = Entry.copyOf(value);
    store.put(key, next);
    return new ConditionalPutResult(true, next.etag);
  }

  // --- internals ---

  private static boolean etagEquals(String current, String expected) {
    if (expected == null) return current == null; // "must not exist"
    return expected.equals(current);
  }

  private static String computeEtag(byte[] value) {
    try {
      byte[] digest = MessageDigest.getInstance("SHA-256").digest(value);
      StringBuilder hex = new StringBuilder(digest.length * 2);
      for (byte b : digest) {
        hex.append(Character.forDigit((b >>> 4) & 0xF, 16));
        hex.append(Character.forDigit(b & 0xF, 16));
      }
      return hex.toString();
    } catch (NoSuchAlgorithmException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  /** Returns a read-only snapshot of the internal store (for diagnostics / tests). */
  Map<String, byte[]> snapshot() {
    return store.entrySet().stream()
        .collect(
            java.util.stream.Collectors.toUnmodifiableMap(
                Map.Entry::getKey, entry -> entry.getValue().valueCopy()));
  }

  private record Entry(byte[] value, String etag) {
    static Entry copyOf(byte[] value) {
      byte[] copy = Arrays.copyOf(value, value.length);
      return new Entry(copy, computeEtag(copy));
    }

    byte[] valueCopy() {
      return Arrays.copyOf(value, value.length);
    }
  }
}
