package com.integrallis.vectors.db;

import com.integrallis.vectors.db.filter.Filter;
import com.integrallis.vectors.db.id.InMemoryIdMapper;
import com.integrallis.vectors.db.index.FlatScanAdapter;
import com.integrallis.vectors.db.index.IndexSpi;
import com.integrallis.vectors.db.internal.StagingBuffer;
import com.integrallis.vectors.db.metadata.InMemoryMetadataStore;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Step 3 in-memory {@link VectorCollection} implementation with a volatile-snapshot publication
 * model.
 *
 * <p><b>Concurrency model.</b> All live state is packed into an immutable {@link Generation} record
 * that is republished atomically via a single volatile write whenever {@link #commit()} runs.
 * Readers ({@link #search}, {@link #get}, {@link #contains}, {@link #size}, {@link #physicalSize})
 * start every call with exactly one volatile read of {@link #generation} and then operate only on
 * that snapshot — no lock is ever acquired on the read path, so commits never block readers and
 * readers never block each other.
 *
 * <p>Writers ({@link #add}, {@link #addAll}, {@link #commit}) serialize through a private {@link
 * ReentrantLock} that does not participate in the read path. Newly-staged documents live in a
 * {@link StagingBuffer} that is itself unsynchronized; the writer lock is its sole guard. On
 * commit, the pipeline builds a fully-populated successor generation (copied id mapper + metadata
 * store, freshly-built SPI) and swaps the volatile pointer in one volatile write. In-flight reads
 * that captured the pre-commit snapshot finish against it normally because none of its state is
 * mutated after publication.
 *
 * <p><b>Step 3 scope.</b> Only {@link com.integrallis.vectors.db.filter.Filter.All} (or {@code
 * null}) is executed; any other filter throws {@link UnsupportedOperationException}. {@code
 * upsert}, {@code delete}, {@code deleteWhere}, and {@code compact} throw {@link
 * UnsupportedOperationException} — they land in later steps.
 */
final class VectorCollectionImpl implements VectorCollection {

  /**
   * Immutable snapshot of the collection's searchable state. Published via a single volatile write
   * from {@link #commit}. Readers capture one volatile reference and use it for the full call —
   * every field inside is independently consistent because the Generation is never mutated after
   * publication.
   */
  private record Generation(
      IndexSpi spi,
      InMemoryIdMapper idMapper,
      InMemoryMetadataStore metadataStore,
      int liveCount) {}

  private final VectorCollectionConfig config;
  private final ReentrantLock writerLock = new ReentrantLock();

  /** Staging buffer, guarded exclusively by {@link #writerLock}. */
  private final StagingBuffer staging = new StagingBuffer();

  /**
   * Current searchable generation. Published via volatile write from {@link #commit}; read from
   * every read path with a single volatile load. Never mutated after publication.
   */
  private volatile Generation generation;

  /**
   * Set to {@code true} by {@link #close()} under {@link #writerLock}. Read from every read path
   * (without lock) to fast-fail late callers. An in-flight reader that captured the pre-close
   * generation may complete normally.
   */
  private volatile boolean closed;

  VectorCollectionImpl(VectorCollectionConfig config) {
    this.config = Objects.requireNonNull(config, "config must not be null");
    IndexSpi emptySpi = new FlatScanAdapter();
    emptySpi.build(new float[0][], config.metric());
    this.generation =
        new Generation(emptySpi, new InMemoryIdMapper(), new InMemoryMetadataStore(), 0);
  }

  @Override
  public void add(Document doc) {
    validateForInsert(doc);
    writerLock.lock();
    try {
      ensureOpen();
      stageUnderLock(doc);
      maybeAutoCommit();
    } finally {
      writerLock.unlock();
    }
  }

  @Override
  public void addAll(Collection<Document> docs) {
    Objects.requireNonNull(docs, "docs must not be null");
    if (docs.isEmpty()) {
      return;
    }
    // Validate every doc OUTSIDE the lock — dimension / null checks don't need the writer lock,
    // and rejecting early on dimension errors avoids partial batch state.
    for (Document d : docs) {
      validateForInsert(d);
    }
    writerLock.lock();
    try {
      ensureOpen();
      for (Document d : docs) {
        stageUnderLock(d);
      }
      maybeAutoCommit();
    } finally {
      writerLock.unlock();
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
   * Must be called under {@link #writerLock}. Rejects duplicate ids (against both the live
   * generation and the staging buffer) in O(1) and appends to staging.
   */
  private void stageUnderLock(Document doc) {
    String id = doc.id();
    // Read live generation without a volatile barrier — we're single-writer under writerLock.
    if (generation.idMapper().contains(id) || staging.contains(id)) {
      throw new IllegalArgumentException("Duplicate id: " + id);
    }
    // append returns false only on staging-local dup, which we already checked.
    staging.append(doc);
  }

  /**
   * Must be called under {@link #writerLock}. If the staging buffer has crossed {@code
   * autoCommitThreshold}, runs an inline commit without releasing the lock.
   */
  private void maybeAutoCommit() {
    if (staging.size() >= config.autoCommitThreshold()) {
      commitUnderLock();
    }
  }

  @Override
  public void commit() {
    writerLock.lock();
    try {
      ensureOpen();
      commitUnderLock();
    } finally {
      writerLock.unlock();
    }
  }

  /**
   * Must be called under {@link #writerLock}. Builds a fully-populated successor {@link Generation}
   * off the write path and publishes it via a single volatile write. Readers continue to see the
   * old generation until the volatile store, then immediately see the new one; readers that
   * captured the old snapshot before the store finish against it normally.
   */
  private void commitUnderLock() {
    if (staging.isEmpty()) {
      return;
    }

    // 1. Read the current generation. Volatile semantics are not required here because we hold
    //    writerLock (sole mutator) — but the read is still safe either way.
    Generation oldGen = this.generation;
    int liveCount = oldGen.liveCount();
    int stagedCount = staging.size();
    int newSize = liveCount + stagedCount;

    // 2. Copy the id mapper and metadata store so the successor is fully independent of oldGen.
    InMemoryIdMapper newMapper = InMemoryIdMapper.copyOf(oldGen.idMapper());
    InMemoryMetadataStore newMeta = InMemoryMetadataStore.copyOf(oldGen.metadataStore());

    // 3. Reconstruct the full dense vector matrix. Reuse the existing float[] references from the
    //    old metadata store for the live prefix (immutable-by-convention per Document Javadoc),
    //    then append staged vectors in order.
    float[][] next = new float[newSize][];
    for (int i = 0; i < liveCount; i++) {
      Document stored = oldGen.metadataStore().get(i);
      if (stored == null) {
        throw new IllegalStateException(
            "Missing document in metadata store for ordinal " + i + " during commit");
      }
      next[i] = stored.vector();
    }

    List<Document> stagedDocs = staging.documents();
    for (int i = 0; i < stagedCount; i++) {
      Document doc = stagedDocs.get(i);
      int ordinal = newMapper.put(doc.id());
      int expectedOrdinal = liveCount + i;
      if (ordinal != expectedOrdinal) {
        throw new IllegalStateException(
            "Ordinal mismatch: expected " + expectedOrdinal + " but got " + ordinal);
      }
      newMeta.put(ordinal, doc);
      next[ordinal] = doc.vector();
    }

    // 4. Build a fresh SPI from scratch. We do this WITHOUT touching oldGen — readers are still
    //    hitting it through the volatile pointer.
    IndexSpi newSpi = new FlatScanAdapter();
    newSpi.build(next, config.metric());

    // 5. Publish the new generation atomically via a single volatile write. Every subsequent
    //    volatile read of `generation` sees the new record; any in-flight reader finishes its
    //    search against the (still-live, unmutated) old record.
    this.generation = new Generation(newSpi, newMapper, newMeta, newSize);
    staging.clear();

    // 6. Close the old SPI. Flat scan is a no-op close; this is the hook for Step 4's mmap-backed
    //    backends where reader ref-counting will move around. Closing after publication is safe
    //    for flat scan because the old `float[][]` is no longer referenced from the facade — only
    //    from in-flight search stacks, which have their own loop-local scope.
    try {
      oldGen.spi().close();
    } catch (Exception e) {
      throw new RuntimeException("Failed to close previous index SPI", e);
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
    Filter filter = request.filter();
    if (filter != null && !(filter instanceof Filter.All)) {
      throw new UnsupportedOperationException("filter execution deferred to Step 5");
    }

    // Single volatile read captures the snapshot for the whole call. No lock is taken.
    Generation gen = this.generation;
    ensureOpen();

    long start = System.nanoTime();
    IndexSpi.SearchOutcome outcome =
        gen.spi()
            .search(
                request.query(), request.k(), request.searchListSize(), request.overQueryFactor());

    int[] ordinals = outcome.ordinals();
    float[] scores = outcome.scores();

    List<SearchResult.Hit> hits = new ArrayList<>(ordinals.length);
    for (int i = 0; i < ordinals.length; i++) {
      float score = scores[i];
      if (score < request.minScore()) {
        continue;
      }
      Document stored = gen.metadataStore().get(ordinals[i]);
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
  }

  @Override
  public Document get(String id) {
    Objects.requireNonNull(id, "id must not be null");
    Generation gen = this.generation;
    ensureOpen();
    int ord = gen.idMapper().ordinalOf(id);
    return ord < 0 ? null : gen.metadataStore().get(ord);
  }

  @Override
  public boolean contains(String id) {
    Objects.requireNonNull(id, "id must not be null");
    Generation gen = this.generation;
    ensureOpen();
    return gen.idMapper().contains(id);
  }

  @Override
  public int size() {
    Generation gen = this.generation;
    ensureOpen();
    return gen.liveCount();
  }

  @Override
  public int physicalSize() {
    Generation gen = this.generation;
    ensureOpen();
    return gen.liveCount();
  }

  @Override
  public VectorCollectionConfig config() {
    return config;
  }

  @Override
  public void close() {
    writerLock.lock();
    try {
      if (closed) {
        return;
      }
      closed = true;
      staging.clear();
      Generation oldGen = this.generation;
      if (oldGen != null) {
        try {
          oldGen.spi().close();
        } catch (Exception e) {
          throw new RuntimeException("Failed to close index SPI", e);
        }
      }
    } finally {
      writerLock.unlock();
    }
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("VectorCollection is closed");
    }
  }
}
