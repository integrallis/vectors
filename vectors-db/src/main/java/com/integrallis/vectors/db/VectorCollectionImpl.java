package com.integrallis.vectors.db;

import com.integrallis.vectors.db.filter.Filter;
import com.integrallis.vectors.db.id.InMemoryIdMapper;
import com.integrallis.vectors.db.index.FlatScanAdapter;
import com.integrallis.vectors.db.index.IndexSpi;
import com.integrallis.vectors.db.metadata.InMemoryMetadataStore;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Step 2 in-memory {@link VectorCollection} implementation.
 *
 * <p>Keeps three pieces of state under a {@link ReentrantReadWriteLock}:
 *
 * <ul>
 *   <li>A live {@link IndexSpi} (currently {@link FlatScanAdapter}) over the last committed
 *       generation's vectors.
 *   <li>A {@link InMemoryIdMapper} carrying the external id ↔ dense ordinal bijection.
 *   <li>A {@link InMemoryMetadataStore} mapping ordinals to full {@link Document} records.
 * </ul>
 *
 * <p>{@link #add(Document)} validates and appends to a staging buffer under the write lock but does
 * not touch the SPI. {@link #commit()} concatenates live + staged vectors into a new {@code
 * float[][]}, rebuilds the SPI from scratch, swaps it in, and clears staging — all under the write
 * lock so readers always observe a consistent generation. A committed reader sees the new
 * generation on its next volatile-free read because the read lock forces a memory barrier.
 *
 * <p>Step 2 filter support: only {@link com.integrallis.vectors.db.filter.Filter.All} (or {@code
 * null}) is executed; any other filter throws {@link UnsupportedOperationException}.
 */
final class VectorCollectionImpl implements VectorCollection {

  private final VectorCollectionConfig config;
  private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

  private final InMemoryIdMapper idMapper = new InMemoryIdMapper();
  private final InMemoryMetadataStore metadataStore = new InMemoryMetadataStore();

  // Live vectors in ordinal order: index i → the vector for ordinal i in the current generation.
  private final List<float[]> liveVectors = new ArrayList<>();
  // Documents staged since the last commit, in insertion order.
  private final List<Document> staging = new ArrayList<>();
  // Shadow set of staged ids for O(1) duplicate detection. Must be kept in sync with staging.
  private final Set<String> stagingIds = new HashSet<>();

  private IndexSpi spi;
  private boolean closed;

  VectorCollectionImpl(VectorCollectionConfig config) {
    this.config = Objects.requireNonNull(config, "config must not be null");
    this.spi = new FlatScanAdapter();
    this.spi.build(new float[0][], config.metric());
  }

  @Override
  public void add(Document doc) {
    validateForInsert(doc);
    lock.writeLock().lock();
    try {
      ensureOpen();
      checkAndStage(doc);
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public void addAll(Collection<Document> docs) {
    Objects.requireNonNull(docs, "docs must not be null");
    if (docs.isEmpty()) {
      return;
    }
    // Validate every doc OUTSIDE the lock — dimension / null checks don't need the write lock,
    // and rejecting early on dimension errors avoids partial batch state.
    for (Document d : docs) {
      validateForInsert(d);
    }
    // Single lock acquisition for the whole batch to avoid N-way lock churn.
    lock.writeLock().lock();
    try {
      ensureOpen();
      for (Document d : docs) {
        checkAndStage(d);
      }
    } finally {
      lock.writeLock().unlock();
    }
  }

  /** Lock-free validation: null check, vector present, dimension matches. */
  private void validateForInsert(Document doc) {
    Objects.requireNonNull(doc, "doc must not be null");
    if (doc.vector() == null) {
      throw new IllegalArgumentException("Document vector must not be null on insertion");
    }
    if (doc.vector().length != config.dimension()) {
      throw new IllegalArgumentException(
          "Document vector dimension "
              + doc.vector().length
              + " does not match collection dimension "
              + config.dimension());
    }
  }

  /**
   * Must be called under the write lock. Rejects duplicate ids (against both live and staged sets)
   * in O(1) via the {@link #stagingIds} shadow set, then appends to staging.
   */
  private void checkAndStage(Document doc) {
    String id = doc.id();
    if (idMapper.contains(id) || !stagingIds.add(id)) {
      throw new IllegalArgumentException("Duplicate id: " + id);
    }
    staging.add(doc);
  }

  @Override
  public void commit() {
    lock.writeLock().lock();
    try {
      ensureOpen();
      if (staging.isEmpty()) {
        return;
      }

      // Concatenate live + staged vectors into a new dense array in ordinal order.
      int liveCount = liveVectors.size();
      int newSize = liveCount + staging.size();
      float[][] next = new float[newSize][];
      for (int i = 0; i < liveCount; i++) {
        next[i] = liveVectors.get(i);
      }
      for (int i = 0; i < staging.size(); i++) {
        Document doc = staging.get(i);
        int ordinal = idMapper.put(doc.id());
        // Sanity check: ordinals are handed out sequentially, so they must line up with the row
        // index in the newly built vector matrix.
        int expectedOrdinal = liveCount + i;
        if (ordinal != expectedOrdinal) {
          throw new IllegalStateException(
              "Ordinal mismatch: expected " + expectedOrdinal + " but got " + ordinal);
        }
        metadataStore.put(ordinal, doc);
        next[ordinal] = doc.vector();
        liveVectors.add(doc.vector());
      }

      // Build a fresh SPI from scratch and atomically swap it in. Closing the previous SPI is a
      // no-op for flat scan but is the right hook for future backends with owned resources.
      IndexSpi oldSpi = this.spi;
      IndexSpi newSpi = new FlatScanAdapter();
      newSpi.build(next, config.metric());
      this.spi = newSpi;
      staging.clear();
      stagingIds.clear();
      try {
        oldSpi.close();
      } catch (Exception e) {
        throw new RuntimeException("Failed to close previous index SPI", e);
      }
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public SearchResult search(SearchRequest request) {
    Objects.requireNonNull(request, "request must not be null");
    if (request.query().length != config.dimension()) {
      throw new IllegalArgumentException(
          "Query dimension "
              + request.query().length
              + " does not match collection dimension "
              + config.dimension());
    }

    // Step 2: only Filter.All (or null) is honored.
    Filter filter = request.filter();
    if (filter != null && !(filter instanceof Filter.All)) {
      throw new UnsupportedOperationException("filter execution deferred to Step 5");
    }

    lock.readLock().lock();
    try {
      ensureOpen();
      // Time the full search path: SPI traversal + metadata hydration + projection. This is the
      // end-to-end latency the caller observes for a single search() call.
      long start = System.nanoTime();
      IndexSpi.SearchOutcome outcome =
          spi.search(
              request.query(), request.k(), request.searchListSize(), request.overQueryFactor());

      int[] ordinals = outcome.ordinals();
      float[] scores = outcome.scores();

      List<SearchResult.Hit> hits = new ArrayList<>(ordinals.length);
      for (int i = 0; i < ordinals.length; i++) {
        float score = scores[i];
        if (score < request.minScore()) {
          continue;
        }
        Document stored = metadataStore.get(ordinals[i]);
        if (stored == null) {
          continue;
        }
        Document projected =
            new Document(
                stored.id(),
                request.includeVector() ? stored.vector() : null,
                request.includeText() ? stored.text() : null,
                request.includeMetadata() ? stored.metadata() : null);
        hits.add(new SearchResult.Hit(stored.id(), score, projected));
      }
      long elapsed = System.nanoTime() - start;
      return new SearchResult(hits, elapsed);
    } finally {
      lock.readLock().unlock();
    }
  }

  @Override
  public Document get(String id) {
    Objects.requireNonNull(id, "id must not be null");
    lock.readLock().lock();
    try {
      ensureOpen();
      int ord = idMapper.ordinalOf(id);
      return ord < 0 ? null : metadataStore.get(ord);
    } finally {
      lock.readLock().unlock();
    }
  }

  @Override
  public boolean contains(String id) {
    Objects.requireNonNull(id, "id must not be null");
    lock.readLock().lock();
    try {
      ensureOpen();
      return idMapper.contains(id);
    } finally {
      lock.readLock().unlock();
    }
  }

  @Override
  public int size() {
    lock.readLock().lock();
    try {
      ensureOpen();
      return liveVectors.size();
    } finally {
      lock.readLock().unlock();
    }
  }

  @Override
  public int physicalSize() {
    lock.readLock().lock();
    try {
      ensureOpen();
      return liveVectors.size();
    } finally {
      lock.readLock().unlock();
    }
  }

  @Override
  public VectorCollectionConfig config() {
    return config;
  }

  @Override
  public void close() {
    lock.writeLock().lock();
    try {
      if (closed) {
        return;
      }
      closed = true;
      staging.clear();
      stagingIds.clear();
      liveVectors.clear();
      idMapper.clear();
      metadataStore.clear();
      if (spi != null) {
        try {
          spi.close();
        } catch (Exception e) {
          throw new RuntimeException("Failed to close index SPI", e);
        }
        spi = null;
      }
    } finally {
      lock.writeLock().unlock();
    }
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("VectorCollection is closed");
    }
  }
}
