package com.integrallis.vectors.db;

import com.integrallis.vectors.db.filter.Filter;
import com.integrallis.vectors.db.id.IdMapper;
import com.integrallis.vectors.db.id.InMemoryIdMapper;
import com.integrallis.vectors.db.index.FlatScanAdapter;
import com.integrallis.vectors.db.index.HnswIndexAdapter;
import com.integrallis.vectors.db.index.IndexSpi;
import com.integrallis.vectors.db.index.MappedFlatScanAdapter;
import com.integrallis.vectors.db.internal.StagingBuffer;
import com.integrallis.vectors.db.metadata.InMemoryMetadataStore;
import com.integrallis.vectors.db.metadata.MetadataStore;
import com.integrallis.vectors.db.storage.Checksums;
import com.integrallis.vectors.db.storage.FileFormat;
import com.integrallis.vectors.db.storage.GenerationDirectory;
import com.integrallis.vectors.db.storage.Manifest;
import com.integrallis.vectors.db.storage.MappedIdMapper;
import com.integrallis.vectors.db.storage.MappedMetadataStore;
import com.integrallis.vectors.db.storage.MemorySegmentVectors;
import com.integrallis.vectors.storage.memory.AlignmentUtil;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Step 4a {@link VectorCollection} implementation with a volatile-snapshot publication model and
 * dual in-memory / mmap-backed persistence.
 *
 * <p><b>Concurrency model.</b> All live state is packed into an immutable {@link Generation} (plain
 * class, not record, so it can carry an {@link AtomicInteger} refcount). On every read path the
 * caller:
 *
 * <ol>
 *   <li>does a single volatile read of {@link #generation}
 *   <li>acquires a refcount on that snapshot ({@link Generation#acquire()})
 *   <li>uses the snapshot for the entire call
 *   <li>releases the refcount in a {@code finally} block
 * </ol>
 *
 * This guarantees that even when {@link #commit()} or {@link #close()} retires a generation, any
 * in-flight reader that already acquired a refcount on it finishes normally against unmutated
 * state. Once the facade's handle <i>and</i> every in-flight reader release, the refcount drops to
 * zero and {@link Generation#closeResources()} fires exactly once — closing the SPI and (in
 * persistent mode) unmapping the shared {@link Arena} that owns {@code vectors.bin}, {@code
 * idmap.bin}, and {@code metadata.bin} atomically.
 *
 * <p>Writers ({@link #add}, {@link #addAll}, {@link #commit}) serialize through a private {@link
 * ReentrantLock} that does not participate in the read path. Staging is a {@link StagingBuffer}
 * whose sole guard is the writer lock.
 *
 * <p><b>Dual mode.</b> When {@link VectorCollectionConfig#storageRoot()} is {@code null} the
 * collection runs in the Step 3 in-memory model (fresh {@link FlatScanAdapter} rebuild per commit).
 * When non-null, the constructor runs {@link GenerationDirectory#recover} to find or bootstrap a
 * generation, opens its mapped stores through a fresh shared {@link Arena}, and every {@link
 * #commit()} writes a new {@code gen-NNNN/} directory through {@link
 * GenerationDirectory#writeGeneration} before publishing the new snapshot.
 *
 * <p><b>Scope.</b> Only {@link Filter.All} (or {@code null}) is executed; any other filter throws
 * {@link UnsupportedOperationException} (deferred to Step 5). {@code upsert}, {@code delete},
 * {@code deleteWhere}, and {@code compact} throw {@link UnsupportedOperationException} (deferred to
 * Steps 5/6).
 */
final class VectorCollectionImpl implements VectorCollection {

  private static final Logger LOGGER = Logger.getLogger(VectorCollectionImpl.class.getName());

  /**
   * Immutable snapshot of the collection's searchable state plus a per-generation refcount.
   * Published via a single volatile write from {@link #commit}. Readers capture one volatile
   * reference, acquire a refcount on it, and use it for the full call. The refcount starts at 1
   * (the facade's handle); every reader bumps it; every release decrements it; the transition to
   * zero triggers {@link #closeResources()} exactly once.
   */
  private static final class Generation {
    final IndexSpi spi;
    final IdMapper idMapper;
    final MetadataStore metadataStore;
    final int liveCount;
    final long generationNumber;

    /** Shared arena owning all mmap'd files for this generation; {@code null} in in-memory mode. */
    final Arena arena;

    /** Absolute path to the {@code gen-NNNN/} directory; {@code null} in in-memory mode. */
    final Path directory;

    /** Zero-copy view of {@code vectors.bin}; {@code null} in in-memory mode. */
    final MemorySegmentVectors mappedVectors;

    private final AtomicInteger refs = new AtomicInteger(1);

    Generation(
        IndexSpi spi,
        IdMapper idMapper,
        MetadataStore metadataStore,
        int liveCount,
        long generationNumber,
        Arena arena,
        Path directory,
        MemorySegmentVectors mappedVectors) {
      this.spi = spi;
      this.idMapper = idMapper;
      this.metadataStore = metadataStore;
      this.liveCount = liveCount;
      this.generationNumber = generationNumber;
      this.arena = arena;
      this.directory = directory;
      this.mappedVectors = mappedVectors;
    }

    /**
     * Attempts to increment the refcount. Returns {@code false} if this generation has already been
     * retired (refcount dropped to zero).
     */
    boolean acquire() {
      while (true) {
        int c = refs.get();
        if (c <= 0) {
          return false;
        }
        if (refs.compareAndSet(c, c + 1)) {
          return true;
        }
      }
    }

    /** Decrements the refcount. Releases resources exactly once when it reaches zero. */
    void release() {
      int c = refs.decrementAndGet();
      if (c == 0) {
        closeResources();
      }
    }

    private void closeResources() {
      try {
        spi.close();
      } catch (Exception e) {
        // SPI close is a no-op in all current (Step 4a) implementations, but graph-backed SPIs
        // landing in Step 4b may need to release real resources here. Log rather than swallow
        // so the failure is observable, but don't rethrow — we still need to run the arena
        // close below to avoid leaking mmap handles.
        LOGGER.log(
            Level.WARNING,
            e,
            () ->
                "SPI close failed for generation "
                    + generationNumber
                    + "; continuing to"
                    + " release arena");
      }
      if (arena != null) {
        arena.close();
      }
    }
  }

  /**
   * {@link GenerationDirectory.GenerationSource} backed by pre-built byte images for each of the
   * generation's data files. The commit pipeline computes CRC32s before constructing the {@link
   * Manifest}, which requires having the bytes in memory rather than streaming them to disk
   * directly — so this source just replays each byte array into the {@code gen-NNNN.tmp/} directory
   * via {@link MappedIdMapper.Writer#writeBytesAndFsync(Path, byte[])} (the same helper is used for
   * every file since it's just "write + fsync").
   *
   * <p>{@code graphBytes} is {@code null} (or zero-length) for flat-scan generations, in which case
   * {@link #writeGraph(Path)} is a no-op and no {@code graph.bin} entry is written. Graph-backed
   * generations (HNSW) pass the encoded adjacency bytes from {@code HnswGraphCodec.encode(graph)}.
   */
  private static final class BufferedGenerationSource
      implements GenerationDirectory.GenerationSource {
    private final byte[] vectorsBytes;
    private final byte[] idmapBytes;
    private final byte[] metadataBytes;
    private final byte[] graphBytes; // null or empty ⇒ no graph.bin

    BufferedGenerationSource(
        byte[] vectorsBytes, byte[] idmapBytes, byte[] metadataBytes, byte[] graphBytes) {
      this.vectorsBytes = vectorsBytes;
      this.idmapBytes = idmapBytes;
      this.metadataBytes = metadataBytes;
      this.graphBytes = graphBytes;
    }

    @Override
    public void writeVectors(Path destination) throws IOException {
      MappedIdMapper.Writer.writeBytesAndFsync(destination, vectorsBytes);
    }

    @Override
    public void writeIdmap(Path destination) throws IOException {
      MappedIdMapper.Writer.writeBytesAndFsync(destination, idmapBytes);
    }

    @Override
    public void writeMetadata(Path destination) throws IOException {
      MappedMetadataStore.Writer.writeBytesAndFsync(destination, metadataBytes);
    }

    @Override
    public void writeGraph(Path destination) throws IOException {
      // GenerationDirectory only calls this for graph index types; a null or empty buffer
      // in that context is a programmer error (the manifest's graphBinLength would be 0 while
      // the caller declared HNSW). Guard defensively so the tmp dir cleanup path still fires.
      if (graphBytes == null || graphBytes.length == 0) {
        throw new IOException(
            "BufferedGenerationSource.writeGraph invoked with no graph bytes (did the caller"
                + " forget to pass HnswGraphCodec.encode(graph) through the constructor?)");
      }
      MappedIdMapper.Writer.writeBytesAndFsync(destination, graphBytes);
    }
  }

  private final VectorCollectionConfig config;
  private final ReentrantLock writerLock = new ReentrantLock();

  /** Staging buffer, guarded exclusively by {@link #writerLock}. */
  private final StagingBuffer staging = new StagingBuffer();

  /**
   * Current searchable generation; {@code null} once the collection is closed. Published via
   * volatile write from {@link #commit} and {@link #close}.
   */
  private volatile Generation generation;

  /**
   * Next generation number to write. Guarded by {@link #writerLock}. Advances <i>unconditionally
   * after a successful {@link GenerationDirectory#writeGeneration}</i>, even if the subsequent
   * {@link #openGeneration} fails — so a retry never collides with the already-durable directory.
   * In persistent mode this is bootstrapped to {@code recoveredGen.generationNumber + 1}; in
   * in-memory mode it is unused (and left at 0).
   */
  private long nextGenerationNumber;

  /**
   * Test-only hook. When non-null, {@link #openGeneration} throws this {@link IOException} instead
   * of opening the generation directory. Used by {@code VectorDbPersistenceTest} to exercise the
   * "write succeeds then open fails" recovery path (audit A1) without having to corrupt files on
   * disk. {@code volatile} so test writes are visible to the writer-lock-held commit path.
   */
  volatile IOException openGenerationFailureHook;

  VectorCollectionImpl(VectorCollectionConfig config) {
    this.config = Objects.requireNonNull(config, "config must not be null");
    if (config.storageRoot() == null) {
      this.generation = bootstrapInMemory();
      this.nextGenerationNumber = 0L;
    } else {
      try {
        this.generation = bootstrapPersistent(config.storageRoot());
        this.nextGenerationNumber = this.generation.generationNumber + 1L;
      } catch (IOException e) {
        throw new UncheckedIOException(
            "Failed to open persistent collection at " + config.storageRoot(), e);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Bootstrap (constructor helpers)
  // ---------------------------------------------------------------------------

  private Generation bootstrapInMemory() {
    IndexSpi emptySpi = newInMemoryAdapter();
    emptySpi.build(new float[0][], config.metric());
    return new Generation(
        emptySpi, new InMemoryIdMapper(), new InMemoryMetadataStore(), 0, 0L, null, null, null);
  }

  /**
   * Creates a fresh in-memory {@link IndexSpi} that matches {@code config.indexType()}. Only used
   * by the in-memory code paths (bootstrap and {@link #commitInMemory}). For HNSW the adapter is
   * parameterized from {@code config.hnswParams()} — which is guaranteed non-null by {@link
   * VectorCollectionConfig}'s compact constructor when {@code indexType == HNSW}.
   */
  private IndexSpi newInMemoryAdapter() {
    return switch (config.indexType()) {
      case FLAT -> new FlatScanAdapter();
      case HNSW -> {
        VectorCollectionConfig.HnswParams p = config.hnswParams();
        yield new HnswIndexAdapter(p.m(), p.efConstruction());
      }
      case VAMANA ->
          throw new UnsupportedOperationException(
              "indexType VAMANA deferred to a later step (Step 4c)");
      case IVF_FLAT ->
          throw new UnsupportedOperationException("indexType IVF_FLAT deferred to a later step");
    };
  }

  private Generation bootstrapPersistent(Path storageRoot) throws IOException {
    // Build the bootstrap payload once. The recovery sweep only materializes these bytes when the
    // root is empty; if an existing valid generation is present, recover() opens that instead and
    // the bootstrap arguments are effectively unused.
    byte[] emptyIdmap = MappedIdMapper.Writer.toBytes(List.of());
    byte[] emptyMetadata = MappedMetadataStore.Writer.toBytes(List.of());
    byte[] emptyVectors = new byte[0];
    Manifest bootstrap =
        Manifest.build(
            config,
            0L,
            0L,
            (long) emptyVectors.length,
            Checksums.ofBytes(emptyVectors),
            (long) emptyMetadata.length,
            Checksums.ofBytes(emptyMetadata),
            (long) emptyIdmap.length,
            Checksums.ofBytes(emptyIdmap),
            0L,
            0L);
    GenerationDirectory.GenerationSource bootstrapSource =
        new BufferedGenerationSource(emptyVectors, emptyIdmap, emptyMetadata, null);

    GenerationDirectory.RecoveryResult rr =
        GenerationDirectory.recover(storageRoot, bootstrapSource, bootstrap);
    return openGeneration(rr.generationDir(), rr.manifest());
  }

  /**
   * Opens the four on-disk files for a generation under a fresh shared {@link Arena} and returns a
   * ready-to-use {@link Generation} with refcount initialized to 1 (the facade's handle). On
   * failure anywhere in the open sequence, the arena is closed so the partial state is released.
   */
  private Generation openGeneration(Path genDir, Manifest manifest) throws IOException {
    IOException injected = openGenerationFailureHook;
    if (injected != null) {
      throw injected;
    }
    Arena arena = Arena.ofShared();
    try {
      MemorySegmentVectors mapped =
          MemorySegmentVectors.open(
              genDir.resolve(FileFormat.VECTORS_FILE),
              Math.toIntExact(manifest.liveCount()),
              manifest.dimension(),
              arena);
      IdMapper idMapper = MappedIdMapper.open(genDir.resolve(FileFormat.IDMAP_FILE), arena);
      MetadataStore metadataStore =
          MappedMetadataStore.open(genDir.resolve(FileFormat.METADATA_FILE), arena);
      // TODO Step 4b Phase 5: switch on manifest.indexType() — for IndexType.HNSW, read the
      // graph.bin file from genDir, CRC-verify against manifest.graphBinCrc32(), decode via
      // HnswGraphCodec.decode(), wrap `mapped` as a RandomAccessVectors, and construct a
      // MappedHnswIndexAdapter instead of MappedFlatScanAdapter. The Manifest v2 already carries
      // graphBinLength/graphBinCrc32 and GenerationDirectory already writes graph.bin via the
      // GRAPH_INDEX_TYPES dispatch, so this is the last remaining piece. VectorCollectionBuilder
      // currently rejects the (HNSW, persistent) combination up front so no existing generation
      // can reach this point with manifest.indexType() == HNSW — but that guard will be relaxed
      // as soon as the HNSW branch below is implemented.
      IndexSpi spi = new MappedFlatScanAdapter(mapped, config.metric());
      return new Generation(
          spi,
          idMapper,
          metadataStore,
          Math.toIntExact(manifest.liveCount()),
          manifest.generationNumber(),
          arena,
          genDir,
          mapped);
    } catch (Exception e) {
      try {
        arena.close();
      } catch (Exception suppressed) {
        e.addSuppressed(suppressed);
      }
      if (e instanceof IOException io) {
        throw io;
      }
      throw new IOException("Failed to open generation at " + genDir, e);
    }
  }

  // ---------------------------------------------------------------------------
  // Write API
  // ---------------------------------------------------------------------------

  @Override
  public void add(Document doc) {
    validateForInsert(doc);
    writerLock.lock();
    try {
      ensureOpenUnderLock();
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
    // Validate outside the lock so dimension/null errors don't hold the writer lock and don't
    // leave the staging buffer in a partially-populated state.
    for (Document d : docs) {
      validateForInsert(d);
    }
    writerLock.lock();
    try {
      ensureOpenUnderLock();
      for (Document d : docs) {
        stageUnderLock(d);
      }
      maybeAutoCommit();
    } finally {
      writerLock.unlock();
    }
  }

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
   * Must be called under {@link #writerLock}. Rejects duplicate ids (against the live generation
   * and the staging buffer) in O(1) and appends to staging.
   */
  private void stageUnderLock(Document doc) {
    String id = doc.id();
    // Plain read — we hold the writer lock so nobody else is mutating this.generation.
    if (generation.idMapper.contains(id) || staging.contains(id)) {
      throw new IllegalArgumentException("Duplicate id: " + id);
    }
    staging.append(doc);
  }

  private void maybeAutoCommit() {
    if (staging.size() >= config.autoCommitThreshold()) {
      commitUnderLock();
    }
  }

  @Override
  public void commit() {
    writerLock.lock();
    try {
      ensureOpenUnderLock();
      commitUnderLock();
    } finally {
      writerLock.unlock();
    }
  }

  /**
   * Must be called under {@link #writerLock}. Branches on whether the collection is in-memory or
   * persistent and delegates to the appropriate commit body. Both branches publish via a single
   * volatile write on {@link #generation} and then release the old generation's facade handle.
   */
  private void commitUnderLock() {
    if (staging.isEmpty()) {
      return;
    }
    Generation oldGen = this.generation;
    if (config.storageRoot() == null) {
      commitInMemory(oldGen);
    } else {
      commitPersistent(oldGen);
    }
  }

  // ---------------------------------------------------------------------------
  // In-memory commit (Step 3 semantics preserved)
  // ---------------------------------------------------------------------------

  private void commitInMemory(Generation oldGen) {
    int liveCount = oldGen.liveCount;
    int stagedCount = staging.size();
    int newSize = liveCount + stagedCount;

    InMemoryIdMapper newMapper = InMemoryIdMapper.copyOf((InMemoryIdMapper) oldGen.idMapper);
    InMemoryMetadataStore newMeta =
        InMemoryMetadataStore.copyOf((InMemoryMetadataStore) oldGen.metadataStore);

    // Build the successor vector matrix with DEFENSIVE COPIES of every row. The metadata store
    // retains references to each Document's original vector array (for get(String) hydration),
    // so sharing those arrays with the FlatScanAdapter would let one subsystem silently observe
    // mutations made through the other — even if no current code mutates, the pre-existing Step
    // 3 design would silently corrupt both if a future caller held onto a Document.vector() and
    // mutated it in place. Cloning on commit cuts the alias.
    float[][] next = new float[newSize][];
    for (int i = 0; i < liveCount; i++) {
      Document stored = oldGen.metadataStore.get(i);
      if (stored == null) {
        throw new IllegalStateException(
            "Missing document in metadata store for ordinal " + i + " during commit");
      }
      next[i] = stored.vector().clone();
    }
    List<Document> stagedDocs = staging.documents();
    for (int i = 0; i < stagedCount; i++) {
      Document doc = stagedDocs.get(i);
      int ordinal = newMapper.put(doc.id());
      int expected = liveCount + i;
      if (ordinal != expected) {
        throw new IllegalStateException(
            "Ordinal mismatch: expected " + expected + " but got " + ordinal);
      }
      newMeta.put(ordinal, doc);
      next[ordinal] = doc.vector().clone();
    }

    IndexSpi newSpi = newInMemoryAdapter();
    newSpi.build(next, config.metric());

    Generation newGen = new Generation(newSpi, newMapper, newMeta, newSize, 0L, null, null, null);
    this.generation = newGen;
    staging.clear();
    // Drop the facade's handle on the old generation. Any in-flight reader still holding a
    // refcount on it finishes normally; closeResources() fires on the last release.
    oldGen.release();
  }

  // ---------------------------------------------------------------------------
  // Persistent commit (Step 4a)
  // ---------------------------------------------------------------------------

  private void commitPersistent(Generation oldGen) {
    int liveCount = oldGen.liveCount;
    int stagedCount = staging.size();
    int newSize = liveCount + stagedCount;
    int dim = config.dimension();

    // 1. Build the ordered (id, document) lists for the successor generation.
    List<String> newIds = new ArrayList<>(newSize);
    List<Document> newDocs = new ArrayList<>(newSize);
    for (int i = 0; i < liveCount; i++) {
      newIds.add(oldGen.idMapper.idOf(i));
      newDocs.add(oldGen.metadataStore.get(i));
    }
    for (Document d : staging.documents()) {
      newIds.add(d.id());
      newDocs.add(d);
    }

    // 2. Build the three data-file byte images in memory so we can CRC them before constructing
    //    the manifest (the manifest embeds the CRCs).
    byte[] vectorsBin = buildVectorsBin(oldGen.mappedVectors, liveCount, staging.documents(), dim);
    byte[] idmapBin;
    byte[] metadataBin;
    try {
      idmapBin = MappedIdMapper.Writer.toBytes(newIds);
      metadataBin = MappedMetadataStore.Writer.toBytes(newDocs);
    } catch (IOException e) {
      throw new UncheckedIOException("Failed to serialize commit payload", e);
    }

    // 3. Build the manifest, then hand everything to GenerationDirectory.writeGeneration which
    //    runs the full crash-safe write protocol (tmp dir → files → fsync → rename → CURRENT).
    //    nextGenerationNumber is our monotonic counter — it advances unconditionally after a
    //    successful write (step 3a below) so any subsequent retry after an open-failure lands on
    //    gen-(N+1) instead of colliding with the already-durable gen-N on disk.
    long newGenNumber = nextGenerationNumber;
    Manifest manifest =
        Manifest.build(
            config,
            newGenNumber,
            (long) newSize,
            (long) vectorsBin.length,
            Checksums.ofBytes(vectorsBin),
            (long) metadataBin.length,
            Checksums.ofBytes(metadataBin),
            (long) idmapBin.length,
            Checksums.ofBytes(idmapBin),
            0L,
            0L);

    GenerationDirectory.WriteResult wr;
    try {
      wr =
          GenerationDirectory.writeGeneration(
              config.storageRoot(),
              newGenNumber,
              new BufferedGenerationSource(vectorsBin, idmapBin, metadataBin, null),
              manifest);
    } catch (IOException e) {
      // Write failed — the gen directory was not durably created (GenerationDirectory fsyncs and
      // renames atomically). nextGenerationNumber is NOT advanced; the next retry can reuse the
      // same number.
      throw new UncheckedIOException("Failed to write generation " + newGenNumber, e);
    }

    // 3a. Write succeeded — gen-N and CURRENT are on disk. Advance the counter BEFORE attempting
    //     open so a subsequent open-failure + retry correctly targets gen-(N+1). Crucial for
    //     recovery from transient open failures (fd exhaustion, OOM during mmap, etc.).
    nextGenerationNumber = newGenNumber + 1L;

    // 4. Open the new generation under a fresh shared arena, then publish it atomically via a
    //    single volatile write. Any in-flight reader captured oldGen before the publish and will
    //    finish against it; subsequent reads observe newGen.
    Generation newGen;
    try {
      newGen = openGeneration(wr.generationDir(), wr.manifest());
    } catch (IOException e) {
      // The gen dir and CURRENT pointer are durable on disk, and nextGenerationNumber has been
      // advanced. A subsequent commit() retry will write to gen-(N+1), leaving gen-N as an
      // unreachable orphan (CURRENT will be overwritten). The staging buffer is NOT cleared so
      // the caller can still retry; oldGen stays live so concurrent readers keep observing a
      // consistent (pre-commit) view.
      throw new UncheckedIOException(
          "Generation "
              + newGenNumber
              + " written but cannot be opened; retry commit() to land"
              + " a successor generation",
          e);
    }

    this.generation = newGen;
    staging.clear();
    oldGen.release();
  }

  /**
   * Builds the {@code vectors.bin} byte image for the successor generation. The old generation's
   * bytes are bulk-copied out of the existing mmap'd segment (same stride, same dim) so there is no
   * float-by-float reconstruction; staged vectors are packed afterward as little-endian float32
   * with 64-byte alignment padding between entries, matching {@link MemorySegmentVectors} and
   * {@link com.integrallis.vectors.storage.store.VectorStoreWriter}.
   */
  private static byte[] buildVectorsBin(
      MemorySegmentVectors oldMapped, int liveCount, List<Document> staged, int dim) {
    long strideL = AlignmentUtil.alignUp((long) dim * Float.BYTES, AlignmentUtil.VECTOR_ALIGNMENT);
    if (strideL > Integer.MAX_VALUE) {
      throw new IllegalStateException("vector stride exceeds 2 GiB: " + strideL);
    }
    int stride = (int) strideL;
    long totalL = strideL * (long) (liveCount + staged.size());
    if (totalL > Integer.MAX_VALUE) {
      throw new IllegalStateException("vectors.bin exceeds 2 GiB: " + totalL);
    }
    byte[] out = new byte[(int) totalL];

    // Bulk copy the old generation byte-for-byte. For the very first persistent commit the
    // bootstrap generation has liveCount == 0 so this is a zero-length copy (no-op).
    if (liveCount > 0 && oldMapped != null) {
      long oldBytes = strideL * liveCount;
      MemorySegment.copy(oldMapped.segment(), ValueLayout.JAVA_BYTE, 0L, out, 0, (int) oldBytes);
    }

    // Pack staged vectors starting at the first free stride slot. Alignment padding is already
    // zero from the array initializer.
    int rawVecBytes = dim * Float.BYTES;
    ByteBuffer buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
    buf.position(stride * liveCount);
    for (Document d : staged) {
      float[] v = d.vector();
      for (int j = 0; j < dim; j++) {
        buf.putFloat(v[j]);
      }
      buf.position(buf.position() + (stride - rawVecBytes));
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // Read API
  // ---------------------------------------------------------------------------

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

    Generation gen = acquireReadSnapshot();
    try {
      long start = System.nanoTime();
      IndexSpi.SearchOutcome outcome =
          gen.spi.search(
              request.query(), request.k(), request.searchListSize(), request.overQueryFactor());

      int[] ordinals = outcome.ordinals();
      float[] scores = outcome.scores();

      List<SearchResult.Hit> hits = new ArrayList<>(ordinals.length);
      for (int i = 0; i < ordinals.length; i++) {
        float score = scores[i];
        if (score < request.minScore()) {
          continue;
        }
        Document stored = gen.metadataStore.get(ordinals[i]);
        if (stored == null) {
          continue;
        }
        float[] vector = null;
        if (request.includeVector()) {
          vector = stored.vector();
          // In persistent mode the mapped metadata store returns Documents with vector == null.
          // Hydrate from the mmap'd vectors segment on demand so the caller gets the full row.
          if (vector == null && gen.mappedVectors != null) {
            vector = new float[config.dimension()];
            MemorySegment.copy(
                gen.mappedVectors.vectorSlice(ordinals[i]),
                ValueLayout.JAVA_FLOAT,
                0L,
                vector,
                0,
                config.dimension());
          }
        }
        Document projected =
            new Document(
                stored.id(),
                vector,
                request.includeText() ? stored.text() : null,
                request.includeMetadata() ? stored.metadata() : null);
        hits.add(new SearchResult.Hit(stored.id(), score, projected));
      }
      long elapsed = System.nanoTime() - start;
      return new SearchResult(hits, elapsed);
    } finally {
      gen.release();
    }
  }

  @Override
  public Document get(String id) {
    Objects.requireNonNull(id, "id must not be null");
    Generation gen = acquireReadSnapshot();
    try {
      int ord = gen.idMapper.ordinalOf(id);
      return ord < 0 ? null : gen.metadataStore.get(ord);
    } finally {
      gen.release();
    }
  }

  @Override
  public boolean contains(String id) {
    Objects.requireNonNull(id, "id must not be null");
    Generation gen = acquireReadSnapshot();
    try {
      return gen.idMapper.contains(id);
    } finally {
      gen.release();
    }
  }

  @Override
  public int size() {
    Generation gen = acquireReadSnapshot();
    try {
      return gen.liveCount;
    } finally {
      gen.release();
    }
  }

  @Override
  public int physicalSize() {
    return size();
  }

  @Override
  public VectorCollectionConfig config() {
    return config;
  }

  @Override
  public void flush() {
    // No-op in both modes. The persistent commit pipeline already fsyncs every data file and the
    // manifest before returning, and the mmap'd files are read-only once opened — there is
    // nothing dirty to flush between commits. In-memory mode has nothing to persist.
  }

  @Override
  public void close() {
    writerLock.lock();
    try {
      Generation gen = this.generation;
      if (gen == null) {
        return;
      }
      this.generation = null;
      staging.clear();
      // Drop the facade's handle. Any in-flight reader still holding a refcount finishes against
      // the captured snapshot; closeResources() fires on the last release.
      gen.release();
    } finally {
      writerLock.unlock();
    }
  }

  /**
   * Captures the current generation via a volatile read and acquires a refcount on it. The caller
   * MUST release the snapshot in a {@code finally} block.
   *
   * <p>Retries if the captured generation was retired between the volatile read and the refcount
   * CAS. This window exists because the commit path publishes the new generation via a volatile
   * write and THEN releases the old generation's facade handle — a reader that reads {@code
   * this.generation} just before the volatile publish can observe a still-alive old generation,
   * only to have that generation's refcount drop to zero moments later when the committing writer
   * decrements the facade handle. In that case the reader's {@code acquire()} CAS sees {@code refs
   * == 0} and must re-read {@link #generation}, which is now guaranteed to point at the new
   * generation (or {@code null} if the collection was actually closed). Without this retry the
   * reader would spuriously observe {@code IllegalStateException("closed")} during a concurrent
   * commit, which manifests as the {@code MonotonicObservedSize} flake.
   *
   * @throws IllegalStateException if the collection has actually been closed (generation is null)
   */
  private Generation acquireReadSnapshot() {
    while (true) {
      Generation gen = this.generation;
      if (gen == null) {
        throw new IllegalStateException("VectorCollection is closed");
      }
      if (gen.acquire()) {
        return gen;
      }
      // The generation was retired between the volatile read and the CAS — a newer generation has
      // been published (or close() is racing). Retry the volatile read; on the next pass we either
      // acquire the successor or observe the null and throw.
    }
  }

  /**
   * Must be called under {@link #writerLock}. Fast-path "closed" check for write operations; no
   * refcount bump because writers never read from the generation past this point — they either
   * mutate {@link #staging} (visible only through a subsequent commit) or rebuild the generation
   * wholesale.
   */
  private void ensureOpenUnderLock() {
    if (this.generation == null) {
      throw new IllegalStateException("VectorCollection is closed");
    }
  }
}
