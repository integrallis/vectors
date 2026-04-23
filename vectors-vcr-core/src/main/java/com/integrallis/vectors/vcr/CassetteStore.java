package com.integrallis.vectors.vcr;

import java.util.List;
import java.util.Optional;

/**
 * Abstraction over cassette persistence. Two built-in implementations:
 *
 * <ul>
 *   <li>{@link ExactCassetteStore} — exact key lookup backed by a {@link
 *       com.integrallis.vectors.storage.backend.StorageBackend}
 *   <li>{@code SemanticCassetteStore} (in the {@code vectors-vcr-semantic-db} module) — fuzzy
 *       lookup by vector similarity, with exact-key fallback
 * </ul>
 *
 * <p>Implementations must be safe for concurrent use from a single test process.
 */
public interface CassetteStore extends AutoCloseable {

  /**
   * Persists a cassette under {@code key}, overwriting any previous value.
   *
   * @param key the cassette key
   * @param record the record payload (must not be null)
   */
  void store(CassetteKey key, CassetteRecord record);

  /**
   * Retrieves a cassette by exact key.
   *
   * @param key the cassette key
   * @return the record, or empty if not found
   */
  Optional<CassetteRecord> retrieve(CassetteKey key);

  /**
   * @return {@code true} if a cassette with the given key exists
   */
  boolean exists(CassetteKey key);

  /**
   * Removes a cassette by key. No-op if absent.
   *
   * @param key the cassette key
   */
  void delete(CassetteKey key);

  /**
   * Lists cassette keys for a given test id. The list is not ordered.
   *
   * @param testId the test identifier
   * @return the cassette keys associated with {@code testId}
   */
  List<CassetteKey> listByTestId(String testId);

  /**
   * Flushes any pending writes to the underlying medium. Default is no-op.
   *
   * @throws java.io.IOException on flush failure
   */
  default void flush() throws java.io.IOException {
    // implementations override if buffered
  }

  /**
   * Closes the store and releases resources. Default is no-op.
   *
   * @throws java.io.IOException on close failure
   */
  @Override
  default void close() throws java.io.IOException {
    // implementations override if they hold resources
  }
}
