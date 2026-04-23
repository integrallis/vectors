package com.integrallis.vectors.vcr;

import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Exact-key {@link CassetteStore} implementation backed by a {@link StorageBackend}.
 *
 * <p>Keys are serialized via {@link CassetteKey#serializedKey()}; values are produced by the
 * configured {@link CassetteSerializer}. This is the behavioural analog of the previous Redis-JSON
 * cassette store: exact hits only, no similarity lookup.
 */
public final class ExactCassetteStore implements CassetteStore {

  private final StorageBackend backend;
  private final CassetteSerializer serializer;

  /**
   * Creates a store.
   *
   * @param backend the underlying storage backend
   * @param serializer the serializer used to encode cassette records
   */
  public ExactCassetteStore(StorageBackend backend, CassetteSerializer serializer) {
    this.backend = Objects.requireNonNull(backend, "backend");
    this.serializer = Objects.requireNonNull(serializer, "serializer");
  }

  /** Convenience constructor using the SPI-loaded default serializer. */
  public ExactCassetteStore(StorageBackend backend) {
    this(backend, CassetteSerializer.load());
  }

  @Override
  public void store(CassetteKey key, CassetteRecord record) {
    Objects.requireNonNull(key, "key");
    Objects.requireNonNull(record, "record");
    try {
      backend.put(key.serializedKey(), serializer.serialize(record));
    } catch (IOException e) {
      throw new UncheckedIOException("VCR cassette store failed for " + key.serializedKey(), e);
    }
  }

  @Override
  public Optional<CassetteRecord> retrieve(CassetteKey key) {
    Objects.requireNonNull(key, "key");
    try {
      byte[] bytes = backend.get(key.serializedKey());
      if (bytes == null) {
        return Optional.empty();
      }
      return Optional.of(serializer.deserialize(bytes));
    } catch (IOException e) {
      throw new UncheckedIOException("VCR cassette read failed for " + key.serializedKey(), e);
    }
  }

  @Override
  public boolean exists(CassetteKey key) {
    Objects.requireNonNull(key, "key");
    try {
      return backend.get(key.serializedKey()) != null;
    } catch (IOException e) {
      throw new UncheckedIOException(
          "VCR cassette exists check failed for " + key.serializedKey(), e);
    }
  }

  @Override
  public void delete(CassetteKey key) {
    Objects.requireNonNull(key, "key");
    try {
      backend.delete(key.serializedKey());
    } catch (IOException e) {
      throw new UncheckedIOException("VCR cassette delete failed for " + key.serializedKey(), e);
    }
  }

  @Override
  public List<CassetteKey> listByTestId(String testId) {
    Objects.requireNonNull(testId, "testId");
    List<CassetteKey> result = new ArrayList<>();
    try {
      for (String serialized : backend.list(CassetteKey.KEY_PREFIX + ":")) {
        CassetteKey parsed = CassetteKey.parse(serialized);
        if (parsed != null && parsed.testId().equals(testId)) {
          result.add(parsed);
        }
      }
    } catch (IOException e) {
      throw new UncheckedIOException("VCR cassette list failed for testId=" + testId, e);
    }
    return result;
  }

  /**
   * @return the backing storage (package-private for diagnostics/tests)
   */
  StorageBackend backend() {
    return backend;
  }

  /**
   * @return the serializer (package-private for diagnostics/tests)
   */
  CassetteSerializer serializer() {
    return serializer;
  }
}
