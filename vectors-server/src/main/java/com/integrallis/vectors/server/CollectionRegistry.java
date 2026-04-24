package com.integrallis.vectors.server;

import com.integrallis.vectors.db.VectorCollection;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Thread-safe registry of {@link VectorCollection} instances, keyed by URL-safe collection name.
 *
 * <p>Reads (lookups, listings) hold no registry-level lock and are fully parallel. Mutations
 * (create, drop) acquire a per-name {@link ReentrantLock} so concurrent operations on different
 * collections do not serialize against each other.
 *
 * <p>Per-name locks are retained for the lifetime of the registry (never removed from the map).
 * This prevents a race where one thread removes the lock while another thread holds a stale
 * reference, breaking the mutual-exclusion invariant. The memory overhead is negligible — each lock
 * is ~48 bytes, and the cardinality is bounded by the number of collections ever created.
 */
public final class CollectionRegistry implements AutoCloseable {

  private static final Logger LOG = LoggerFactory.getLogger(CollectionRegistry.class);

  private final ConcurrentHashMap<String, VectorCollection> collections = new ConcurrentHashMap<>();
  private final ConcurrentHashMap<String, ReentrantLock> nameLocks = new ConcurrentHashMap<>();

  /**
   * @return a snapshot view of all currently-managed collection names (no lock held during
   *     iteration by callers)
   */
  public Collection<String> names() {
    return Collections.unmodifiableSet(collections.keySet());
  }

  /**
   * @param name collection name
   * @return the collection if present
   */
  public Optional<VectorCollection> get(String name) {
    return Optional.ofNullable(collections.get(Objects.requireNonNull(name, "name")));
  }

  /**
   * Creates a collection if it does not already exist. The {@code factory} is invoked at most once
   * per name under a per-name lock.
   *
   * @param name collection name
   * @param factory supplier that constructs the collection (typically {@code
   *     VectorCollection.builder()...build()})
   * @return the existing or newly-created collection
   * @throws IllegalStateException if a collection with this name already exists
   */
  public VectorCollection create(String name, Function<String, VectorCollection> factory) {
    Objects.requireNonNull(name, "name");
    Objects.requireNonNull(factory, "factory");
    ReentrantLock lock = nameLocks.computeIfAbsent(name, k -> new ReentrantLock());
    lock.lock();
    try {
      if (collections.containsKey(name)) {
        throw new IllegalStateException("collection already exists: " + name);
      }
      VectorCollection created = factory.apply(name);
      Objects.requireNonNull(created, "factory returned null");
      collections.put(name, created);
      return created;
    } finally {
      lock.unlock();
    }
  }

  /**
   * Drops a collection and closes its underlying resources. Idempotent: returns {@code false} if
   * the name is not registered.
   *
   * @param name collection name
   * @return whether a collection was removed
   */
  public boolean drop(String name) {
    Objects.requireNonNull(name, "name");
    ReentrantLock lock = nameLocks.computeIfAbsent(name, k -> new ReentrantLock());
    lock.lock();
    try {
      VectorCollection removed = collections.remove(name);
      if (removed == null) {
        return false;
      }
      removed.close();
      // Lock intentionally NOT removed from nameLocks — see class Javadoc.
      return true;
    } finally {
      lock.unlock();
    }
  }

  /**
   * @return the number of collections currently registered
   */
  public int size() {
    return collections.size();
  }

  /**
   * @return a live view of the underlying map, for diagnostic tooling only
   */
  public Map<String, VectorCollection> asMap() {
    return Collections.unmodifiableMap(collections);
  }

  @Override
  public void close() {
    for (Map.Entry<String, VectorCollection> e : collections.entrySet()) {
      try {
        e.getValue().close();
      } catch (RuntimeException ex) {
        LOG.warn("Failed to close collection '{}': {}", e.getKey(), ex.getMessage(), ex);
      }
    }
    collections.clear();
    nameLocks.clear();
  }
}
