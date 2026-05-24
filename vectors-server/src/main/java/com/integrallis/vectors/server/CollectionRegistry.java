/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
package com.integrallis.vectors.server;

import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.hybrid.text.TextIndexSpi;
import com.integrallis.vectors.hybrid.text.TextIndexSpiFactory;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
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
  private final ConcurrentHashMap<String, TextIndexSpi> textIndexes = new ConcurrentHashMap<>();
  private final ConcurrentHashMap<String, Instant> createdAt = new ConcurrentHashMap<>();
  private final ConcurrentHashMap<String, AtomicLong> epochs = new ConcurrentHashMap<>();
  private final ConcurrentHashMap<String, ReentrantLock> nameLocks = new ConcurrentHashMap<>();
  private final java.util.concurrent.CopyOnWriteArrayList<Consumer<EpochChange>> listeners =
      new java.util.concurrent.CopyOnWriteArrayList<>();
  private final TextIndexSpiFactory textIndexFactory;
  private volatile Path dataDir;

  /**
   * Creates a registry, discovering a {@link TextIndexSpiFactory} via ServiceLoader if available.
   */
  public CollectionRegistry() {
    TextIndexSpiFactory factory = null;
    for (TextIndexSpiFactory f : ServiceLoader.load(TextIndexSpiFactory.class)) {
      factory = f;
      LOG.info("discovered TextIndexSpiFactory: {}", f.getClass().getName());
      break;
    }
    this.textIndexFactory = factory;
  }

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
      createdAt.put(name, Instant.now());
      epochs.put(name, new AtomicLong(0));
      createTextIndex(name);
      return created;
    } finally {
      lock.unlock();
    }
  }

  /**
   * Registers a previously-persisted collection under its existing name with the supplied creation
   * timestamp. Intended for startup discovery of {@code dataDir} subdirectories — call sites that
   * create a brand-new collection must use {@link #create(String, Function)} instead so the
   * registry-owned {@code Instant.now()} is recorded.
   *
   * @param name collection name
   * @param factory supplier that reopens the persisted collection via {@code
   *     VectorCollection.builder().storagePath(...)...build()}
   * @param createdAt creation timestamp to preserve (typically the filesystem creation time of the
   *     collection directory)
   * @return the reopened collection
   * @throws IllegalStateException if a collection with this name already exists
   */
  public VectorCollection reopen(
      String name,
      Function<String, VectorCollection> factory,
      Instant createdAt,
      long initialEpoch) {
    Objects.requireNonNull(name, "name");
    Objects.requireNonNull(factory, "factory");
    Objects.requireNonNull(createdAt, "createdAt");
    ReentrantLock lock = nameLocks.computeIfAbsent(name, k -> new ReentrantLock());
    lock.lock();
    try {
      if (collections.containsKey(name)) {
        throw new IllegalStateException("collection already exists: " + name);
      }
      VectorCollection reopened = factory.apply(name);
      Objects.requireNonNull(reopened, "factory returned null");
      collections.put(name, reopened);
      this.createdAt.put(name, createdAt);
      epochs.put(name, new AtomicLong(initialEpoch));
      createTextIndex(name);
      return reopened;
    } finally {
      lock.unlock();
    }
  }

  /**
   * @param name collection name
   * @return creation timestamp, or empty if the name is not registered
   */
  public Optional<Instant> createdAt(String name) {
    return Optional.ofNullable(createdAt.get(Objects.requireNonNull(name, "name")));
  }

  /**
   * @param name collection name
   * @return current epoch for this collection, or -1 if unknown
   */
  public long epoch(String name) {
    AtomicLong e = epochs.get(Objects.requireNonNull(name, "name"));
    return e == null ? -1L : e.get();
  }

  /**
   * Bumps the epoch of {@code name} and notifies every registered listener synchronously. Callers
   * invoke this after a successful commit on the underlying {@link VectorCollection}. No-op when
   * the collection is unknown (drop races).
   *
   * @return the new epoch, or -1 if the collection was not found
   */
  public long bumpEpoch(String name) {
    AtomicLong e = epochs.get(Objects.requireNonNull(name, "name"));
    if (e == null) {
      return -1L;
    }
    long next = e.incrementAndGet();
    EpochChange event = new EpochChange(name, next);
    for (Consumer<EpochChange> l : listeners) {
      try {
        l.accept(event);
      } catch (RuntimeException ex) {
        LOG.warn("epoch listener threw: {}", ex.getMessage());
      }
    }
    return next;
  }

  /** Registers a listener for epoch-change events. Returns a handle for removal. */
  public Runnable addEpochListener(Consumer<EpochChange> listener) {
    Objects.requireNonNull(listener, "listener");
    listeners.add(listener);
    return () -> listeners.remove(listener);
  }

  /**
   * Runs {@code action} under the per-name write lock for {@code name}, so a sequence of mutating
   * operations on the same collection (e.g. a batch of {@code upsert}s followed by {@code commit})
   * is observably atomic with respect to other write paths against the same collection.
   * Different-name actions do not block each other.
   *
   * <p>The same {@link ReentrantLock} map used by {@link #create} and {@link #drop} backs this
   * method, so create/drop/write paths all serialise correctly against one another for the same
   * collection name. The lock is reentrant: an action that calls back into {@code
   * runUnderWriteLock} for the same name does not deadlock.
   *
   * <p>Exceptions thrown by {@code action} propagate; the lock is always released. The lock is
   * <i>not</i> created for the collection's lifetime by this call alone — write operations may race
   * with a never-created collection just as they may race with a dropped one; callers must still
   * check {@link #get(String)} inside the action.
   *
   * @param name collection name (non-null)
   * @param action body to run while holding the per-name lock (non-null)
   */
  public void runUnderWriteLock(String name, Runnable action) {
    Objects.requireNonNull(name, "name");
    Objects.requireNonNull(action, "action");
    ReentrantLock lock = nameLocks.computeIfAbsent(name, k -> new ReentrantLock());
    lock.lock();
    try {
      action.run();
    } finally {
      lock.unlock();
    }
  }

  /** Immutable epoch-change notification. */
  public record EpochChange(String name, long epoch) {}

  /**
   * Sets the base data directory for persistent text indexes. When set, text indexes are stored on
   * disk inside each collection's subdirectory. Must be called before any {@link #create} or {@link
   * #reopen} calls.
   *
   * @param dataDir base data directory, or {@code null} for in-memory text indexes
   */
  public void setDataDir(Path dataDir) {
    this.dataDir = dataDir;
  }

  /**
   * @param name collection name
   * @return the text index for this collection, or empty if no TextIndexSpiFactory was discovered
   */
  public Optional<TextIndexSpi> getTextIndex(String name) {
    return Optional.ofNullable(textIndexes.get(Objects.requireNonNull(name, "name")));
  }

  private void createTextIndex(String name) {
    if (textIndexFactory != null) {
      try {
        Path collectionDir = dataDir != null ? dataDir.resolve(name) : null;
        TextIndexSpi index = textIndexFactory.create(name, collectionDir);
        textIndexes.put(name, index);
        LOG.debug("created text index for collection '{}'", name);
      } catch (RuntimeException e) {
        LOG.warn("failed to create text index for '{}': {}", name, e.getMessage());
      }
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
      createdAt.remove(name);
      epochs.remove(name);
      TextIndexSpi ti = textIndexes.remove(name);
      if (ti != null) {
        try {
          ti.drop();
        } catch (RuntimeException e) {
          LOG.warn("failed to drop text index for '{}': {}", name, e.getMessage());
        }
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
    for (Map.Entry<String, TextIndexSpi> e : textIndexes.entrySet()) {
      try {
        e.getValue().close();
      } catch (RuntimeException ex) {
        LOG.warn("Failed to close text index '{}': {}", e.getKey(), ex.getMessage(), ex);
      }
    }
    textIndexes.clear();
    for (Map.Entry<String, VectorCollection> e : collections.entrySet()) {
      try {
        e.getValue().close();
      } catch (RuntimeException ex) {
        LOG.warn("Failed to close collection '{}': {}", e.getKey(), ex.getMessage(), ex);
      }
    }
    collections.clear();
    createdAt.clear();
    epochs.clear();
    nameLocks.clear();
    listeners.clear();
  }
}
