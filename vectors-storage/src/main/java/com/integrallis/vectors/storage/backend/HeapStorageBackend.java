package com.integrallis.vectors.storage.backend;

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

  /** Etag separator — separates hex-content-hash from a monotone counter. */
  private static final String ETAG_SEP = "-";

  private final ConcurrentHashMap<String, byte[]> store = new ConcurrentHashMap<>();
  private final ConcurrentHashMap<String, String> etags = new ConcurrentHashMap<>();

  @Override
  public synchronized void put(String key, byte[] value) {
    store.put(key, Arrays.copyOf(value, value.length));
    etags.put(key, computeEtag(value));
  }

  @Override
  public byte[] get(String key) {
    byte[] v = store.get(key);
    return v == null ? null : Arrays.copyOf(v, v.length);
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
    etags.remove(key);
  }

  @Override
  public synchronized ConditionalPutResult conditionalPut(
      String key, byte[] value, String expectedEtag) {
    String currentEtag = etags.get(key);
    if (!etag_equals(currentEtag, expectedEtag)) {
      return new ConditionalPutResult(false, null);
    }
    String newEtag = computeEtag(value);
    store.put(key, Arrays.copyOf(value, value.length));
    etags.put(key, newEtag);
    return new ConditionalPutResult(true, newEtag);
  }

  // --- internals ---

  private static boolean etag_equals(String current, String expected) {
    if (expected == null) return current == null; // "must not exist"
    return expected.equals(current);
  }

  private static String computeEtag(byte[] value) {
    // CRC32 hex is sufficient for an in-heap etag — not cryptographic.
    java.util.zip.CRC32 crc = new java.util.zip.CRC32();
    crc.update(value);
    return Long.toHexString(crc.getValue()) + ETAG_SEP + value.length;
  }

  /** Returns a read-only snapshot of the internal store (for diagnostics / tests). */
  Map<String, byte[]> snapshot() {
    return Map.copyOf(store);
  }
}
