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
package com.integrallis.vectors.db;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.filter.Filter;
import com.integrallis.vectors.db.cache.QvCache;
import com.integrallis.vectors.db.filter.FilterExecutor;
import com.integrallis.vectors.db.id.IdMapper;
import com.integrallis.vectors.db.id.InMemoryIdMapper;
import com.integrallis.vectors.db.index.CuVsBruteForceAdapter;
import com.integrallis.vectors.db.index.CuVsCagraAdapter;
import com.integrallis.vectors.db.index.FlatScanAdapter;
import com.integrallis.vectors.db.index.HnswIndexAdapter;
import com.integrallis.vectors.db.index.IndexSpi;
import com.integrallis.vectors.db.index.IvfFlatAdapter;
import com.integrallis.vectors.db.index.IvfPqAdapter;
import com.integrallis.vectors.db.index.MappedFlatScanAdapter;
import com.integrallis.vectors.db.index.QuantizedFlatScanAdapter;
import com.integrallis.vectors.db.index.VamanaIndexAdapter;
import com.integrallis.vectors.db.internal.StagingBuffer;
import com.integrallis.vectors.db.metadata.InMemoryMetadataStore;
import com.integrallis.vectors.db.metadata.MetadataStore;
import com.integrallis.vectors.db.storage.Checksums;
import com.integrallis.vectors.db.storage.FileFormat;
import com.integrallis.vectors.db.storage.GenerationDirectory;
import com.integrallis.vectors.db.storage.HnswGraphCodec;
import com.integrallis.vectors.db.storage.Manifest;
import com.integrallis.vectors.db.storage.MappedHnswIndexAdapter;
import com.integrallis.vectors.db.storage.MappedIdMapper;
import com.integrallis.vectors.db.storage.MappedIvfFlatAdapter;
import com.integrallis.vectors.db.storage.MappedIvfPqAdapter;
import com.integrallis.vectors.db.storage.MappedMetadataStore;
import com.integrallis.vectors.db.storage.MappedVamanaIndexAdapter;
import com.integrallis.vectors.db.storage.MemorySegmentRandomAccessVectors;
import com.integrallis.vectors.db.storage.MemorySegmentVectors;
import com.integrallis.vectors.db.storage.QuantizedVectorsCodec;
import com.integrallis.vectors.db.storage.TombstoneCodec;
import com.integrallis.vectors.db.storage.VamanaGraphCodec;
import com.integrallis.vectors.hnsw.HnswGraph;
import com.integrallis.vectors.hnsw.HnswGraphMerger;
import com.integrallis.vectors.ivf.IvfBuildParams;
import com.integrallis.vectors.ivf.IvfIndex;
import com.integrallis.vectors.quantization.ArrayVectorDataset;
import com.integrallis.vectors.quantization.BinaryMode;
import com.integrallis.vectors.quantization.BinaryQuantizer;
import com.integrallis.vectors.quantization.CompressedVectors;
import com.integrallis.vectors.quantization.NVQuantizer;
import com.integrallis.vectors.quantization.ProductQuantizer;
import com.integrallis.vectors.quantization.Quantizer;
import com.integrallis.vectors.quantization.RaBitQuantizer;
import com.integrallis.vectors.quantization.ScalarBits;
import com.integrallis.vectors.quantization.ScalarQuantizer;
import com.integrallis.vectors.quantization.VectorDataset;
import com.integrallis.vectors.storage.memory.AlignmentUtil;
import com.integrallis.vectors.vamana.VamanaGraph;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.IntPredicate;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * {@link VectorCollection} implementation with tombstone-based deletion, upsert, compact, and a
 * volatile-snapshot publication model with dual in-memory / mmap-backed persistence.
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
 * <p><b>Tombstone semantics.</b> Deletes flip a bit in the generation's tombstone {@link BitSet} —
 * no ordinal reuse, no ANN graph modification, no quantized code modification. Tombstoned ordinals
 * retain their original vectors in the ANN index so that graph traversal never operates on
 * undefined zero-vectors (which would produce NaN cosine-similarity scores and corrupt
 * priority-queue ordering). Search results skip tombstoned ordinals via a post-visit tombstone
 * check — not by replacing their vector content. {@link #compact()} is the explicit escape hatch —
 * rebuilds from scratch with dense ordinals.
 */
final class VectorCollectionImpl implements VectorCollection {

  private static final Logger LOGGER = Logger.getLogger(VectorCollectionImpl.class.getName());

  /**
   * Immutable snapshot of the collection's searchable state plus a per-generation refcount.
   * Published via a single volatile write from {@link #commit}. Readers capture one volatile
   * reference, acquire a refcount on it, and use it for the full call.
   */
  private static final class Generation {
    final IndexSpi spi;
    final IdMapper idMapper;
    final MetadataStore metadataStore;
    final int physicalCount;
    final BitSet tombstones;
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
        int physicalCount,
        BitSet tombstones,
        long generationNumber,
        Arena arena,
        Path directory,
        MemorySegmentVectors mappedVectors) {
      this.spi = spi;
      this.idMapper = idMapper;
      this.metadataStore = metadataStore;
      this.physicalCount = physicalCount;
      this.tombstones = tombstones;
      this.generationNumber = generationNumber;
      this.arena = arena;
      this.directory = directory;
      this.mappedVectors = mappedVectors;
    }

    int liveCount() {
      return physicalCount - tombstones.cardinality();
    }

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
   * generation's data files.
   */
  private static final class BufferedGenerationSource
      implements GenerationDirectory.GenerationSource {
    private final byte[] vectorsBytes;
    private final byte[] idmapBytes;
    private final byte[] metadataBytes;
    private final byte[] graphBytes;
    private final byte[] quantizedBytes;
    private final byte[] tombstonesBytes;

    BufferedGenerationSource(
        byte[] vectorsBytes,
        byte[] idmapBytes,
        byte[] metadataBytes,
        byte[] graphBytes,
        byte[] quantizedBytes,
        byte[] tombstonesBytes) {
      this.vectorsBytes = vectorsBytes;
      this.idmapBytes = idmapBytes;
      this.metadataBytes = metadataBytes;
      this.graphBytes = graphBytes;
      this.quantizedBytes = quantizedBytes;
      this.tombstonesBytes = tombstonesBytes;
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
      if (graphBytes == null || graphBytes.length == 0) {
        throw new IOException(
            "BufferedGenerationSource.writeGraph invoked with no graph bytes (did the caller"
                + " forget to pass HnswGraphCodec.encode / VamanaGraphCodec.encode through the"
                + " constructor?)");
      }
      MappedIdMapper.Writer.writeBytesAndFsync(destination, graphBytes);
    }

    @Override
    public void writeQuantized(Path destination) throws IOException {
      if (quantizedBytes == null || quantizedBytes.length == 0) {
        throw new IOException(
            "BufferedGenerationSource.writeQuantized invoked with no quantized bytes (did the"
                + " caller forget to pass QuantizedVectorsCodec.encode through the constructor?)");
      }
      MappedIdMapper.Writer.writeBytesAndFsync(destination, quantizedBytes);
    }

    @Override
    public void writeTombstones(Path destination) throws IOException {
      if (tombstonesBytes == null || tombstonesBytes.length == 0) {
        throw new IOException(
            "BufferedGenerationSource.writeTombstones invoked with no tombstones bytes");
      }
      MappedIdMapper.Writer.writeBytesAndFsync(destination, tombstonesBytes);
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
   * Next generation number to write. Guarded by {@link #writerLock}. Advances unconditionally after
   * a successful write.
   */
  private long nextGenerationNumber;

  /**
   * Test-only hook. When non-null, {@link #openGeneration} throws this {@link IOException} instead
   * of opening the generation directory.
   */
  volatile IOException openGenerationFailureHook;

  /** QVCache — never null; {@link QvCache#DISABLED} when caching is off. */
  private final QvCache queryCache;

  /** Returns the active query cache (test / monitoring access). */
  public QvCache queryCache() {
    return queryCache;
  }

  VectorCollectionImpl(VectorCollectionConfig config) {
    this(config, QvCache.DISABLED);
  }

  VectorCollectionImpl(VectorCollectionConfig config, QvCache cache) {
    this.config = Objects.requireNonNull(config, "config must not be null");
    this.queryCache = Objects.requireNonNull(cache, "cache must not be null");
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
        emptySpi,
        new InMemoryIdMapper(),
        new InMemoryMetadataStore(),
        0,
        new BitSet(),
        0L,
        null,
        null,
        null);
  }

  private IndexSpi newInMemoryAdapter() {
    return switch (config.indexType()) {
      case FLAT -> new FlatScanAdapter();
      case HNSW -> {
        VectorCollectionConfig.HnswParams p = config.hnswParams();
        yield new HnswIndexAdapter(p.m(), p.efConstruction(), p.threads());
      }
      case VAMANA -> {
        VectorCollectionConfig.VamanaParams p = config.vamanaParams();
        yield new VamanaIndexAdapter(
            p.maxDegree(), p.searchListSize(), p.alpha(), p.seed(), p.threads());
      }
      case IVF_FLAT -> {
        VectorCollectionConfig.IvfParams p = config.ivfParams();
        yield new IvfFlatAdapter(p.k(), p.nprobe(), p.maxIter(), p.gamma(), p.soar(), p.seed());
      }
      case IVF_PQ -> {
        VectorCollectionConfig.IvfPqParams p = config.ivfPqParams();
        yield new IvfPqAdapter(
            p.k(),
            p.nprobe(),
            p.maxIter(),
            p.gamma(),
            p.soar(),
            p.seed(),
            p.pqSubspaces(),
            p.pqClusters(),
            p.pqAnisotropicThreshold(),
            p.rescoreFactor());
      }
      case CUVS_BRUTEFORCE ->
          new CuVsBruteForceAdapter(
              (VectorCollectionConfig.CuVsParams.BruteForce) config.cuvsParams());
      case CUVS_CAGRA ->
          new CuVsCagraAdapter((VectorCollectionConfig.CuVsParams.Cagra) config.cuvsParams());
    };
  }

  private Generation bootstrapPersistent(Path storageRoot) throws IOException {
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
            0L,
            0L,
            0L);
    GenerationDirectory.GenerationSource bootstrapSource =
        new BufferedGenerationSource(emptyVectors, emptyIdmap, emptyMetadata, null, null, null);

    GenerationDirectory.RecoveryResult rr =
        GenerationDirectory.recover(storageRoot, bootstrapSource, bootstrap);
    return openGeneration(rr.generationDir(), rr.manifest());
  }

  private Generation openGeneration(Path genDir, Manifest manifest) throws IOException {
    IOException injected = openGenerationFailureHook;
    if (injected != null) {
      throw injected;
    }
    Arena arena = Arena.ofShared();
    try {
      // physicalCount = liveCount + tombstoneCount (liveCount in the manifest is the count of
      // non-tombstoned vectors)
      int physicalCount = Math.toIntExact(manifest.liveCount() + manifest.tombstoneCount());

      MemorySegmentVectors mapped =
          MemorySegmentVectors.open(
              genDir.resolve(FileFormat.VECTORS_FILE), physicalCount, manifest.dimension(), arena);
      IdMapper idMapper = MappedIdMapper.open(genDir.resolve(FileFormat.IDMAP_FILE), arena);
      MetadataStore metadataStore =
          MappedMetadataStore.open(genDir.resolve(FileFormat.METADATA_FILE), arena);

      // Load tombstones if present.
      BitSet tombstones;
      if (manifest.tombstonesBinLength() > 0L) {
        byte[] tombstonesBytes =
            java.nio.file.Files.readAllBytes(genDir.resolve(FileFormat.TOMBSTONES_FILE));
        tombstones = TombstoneCodec.decode(tombstonesBytes);
      } else {
        tombstones = new BitSet();
      }

      IndexSpi spi =
          switch (manifest.indexType()) {
            case FLAT -> new MappedFlatScanAdapter(mapped, config.metric());
            case HNSW ->
                manifest.graphBinLength() > 0L
                    ? openHnswAdapter(genDir, manifest, mapped)
                    : new MappedFlatScanAdapter(mapped, config.metric());
            case VAMANA ->
                manifest.graphBinLength() > 0L
                    ? openVamanaAdapter(genDir, manifest, mapped)
                    : new MappedFlatScanAdapter(mapped, config.metric());
            case IVF_FLAT ->
                manifest.graphBinLength() > 0L
                    ? openIvfFlatAdapter(genDir, manifest, mapped)
                    : new MappedFlatScanAdapter(mapped, config.metric());
            case IVF_PQ ->
                manifest.graphBinLength() > 0L
                    ? openIvfPqAdapter(genDir, manifest, mapped)
                    : new MappedFlatScanAdapter(mapped, config.metric());
            case CUVS_BRUTEFORCE, CUVS_CAGRA ->
                throw new UnsupportedOperationException(
                    "CUVS_* index types do not support persistent storage yet: "
                        + manifest.indexType());
          };

      if (manifest.quantizedBinLength() > 0L) {
        byte[] quantizedBytes =
            java.nio.file.Files.readAllBytes(genDir.resolve(FileFormat.QUANTIZED_FILE));
        try {
          CompressedVectors compressed = QuantizedVectorsCodec.decode(quantizedBytes);
          if (spi instanceof MappedFlatScanAdapter fa) {
            spi = new QuantizedFlatScanAdapter(fa, fa, config.metric(), compressed);
          } else if (spi instanceof MappedHnswIndexAdapter ha) {
            ha.enableQuantization(compressed);
          } else if (spi instanceof MappedVamanaIndexAdapter va) {
            va.enableQuantization(compressed);
          }
        } catch (IOException qe) {
          throw new IOException(
              "Failed to decode quantized.bin for generation " + manifest.generationNumber(), qe);
        }
      }

      return new Generation(
          spi,
          idMapper,
          metadataStore,
          physicalCount,
          tombstones,
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

  private IndexSpi openHnswAdapter(Path genDir, Manifest manifest, MemorySegmentVectors mapped)
      throws IOException {
    Path graphFile = genDir.resolve(FileFormat.GRAPH_FILE);
    if (manifest.graphBinLength() <= 0L) {
      throw new IOException(
          "HNSW generation " + manifest.generationNumber() + " has no graph.bin recorded");
    }
    byte[] graphBytes = java.nio.file.Files.readAllBytes(graphFile);
    HnswGraph graph = HnswGraphCodec.decode(graphBytes);
    MemorySegmentRandomAccessVectors vectors = new MemorySegmentRandomAccessVectors(mapped);
    return new MappedHnswIndexAdapter(graph, vectors, config.metric());
  }

  private IndexSpi openVamanaAdapter(Path genDir, Manifest manifest, MemorySegmentVectors mapped)
      throws IOException {
    Path graphFile = genDir.resolve(FileFormat.GRAPH_FILE);
    if (manifest.graphBinLength() <= 0L) {
      throw new IOException(
          "VAMANA generation " + manifest.generationNumber() + " has no graph.bin recorded");
    }
    byte[] graphBytes = java.nio.file.Files.readAllBytes(graphFile);
    VamanaGraph graph = VamanaGraphCodec.decode(graphBytes);
    MemorySegmentRandomAccessVectors vectors = new MemorySegmentRandomAccessVectors(mapped);
    return new MappedVamanaIndexAdapter(graph, vectors, config.metric());
  }

  private IndexSpi openIvfFlatAdapter(Path genDir, Manifest manifest, MemorySegmentVectors mapped)
      throws IOException {
    if (manifest.graphBinLength() <= 0L) {
      throw new IOException(
          "IVF_FLAT generation " + manifest.generationNumber() + " has no graph.bin recorded");
    }
    byte[] ivfBytes = java.nio.file.Files.readAllBytes(genDir.resolve(FileFormat.GRAPH_FILE));
    // Hydrate the full vector matrix from the mmap'd vectors.bin so IvfIndex can score them.
    int n = (int) (manifest.liveCount() + manifest.tombstoneCount());
    int dim = config.dimension();
    float[][] matrix = new float[n][dim];
    for (int i = 0; i < n; i++) {
      java.lang.foreign.MemorySegment.copy(
          mapped.vectorSlice(i), java.lang.foreign.ValueLayout.JAVA_FLOAT, 0L, matrix[i], 0, dim);
    }
    IvfIndex ivfIndex = IvfIndex.decode(ivfBytes, matrix, config.metric());
    VectorCollectionConfig.IvfParams p = config.ivfParams();
    return new MappedIvfFlatAdapter(ivfIndex, p.nprobe(), p.gamma(), dim);
  }

  private IndexSpi openIvfPqAdapter(Path genDir, Manifest manifest, MemorySegmentVectors mapped)
      throws IOException {
    if (manifest.graphBinLength() <= 0L) {
      throw new IOException(
          "IVF_PQ generation " + manifest.generationNumber() + " has no graph.bin recorded");
    }
    byte[] ivfBytes = java.nio.file.Files.readAllBytes(genDir.resolve(FileFormat.GRAPH_FILE));
    int n = (int) (manifest.liveCount() + manifest.tombstoneCount());
    int dim = config.dimension();
    float[][] matrix = new float[n][dim];
    for (int i = 0; i < n; i++) {
      java.lang.foreign.MemorySegment.copy(
          mapped.vectorSlice(i), java.lang.foreign.ValueLayout.JAVA_FLOAT, 0L, matrix[i], 0, dim);
    }
    IvfIndex ivfIndex = IvfIndex.decode(ivfBytes, matrix, config.metric());
    VectorCollectionConfig.IvfPqParams p = config.ivfPqParams();
    return new MappedIvfPqAdapter(ivfIndex, p.nprobe(), p.gamma(), p.rescoreFactor(), dim);
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

  private void stageUnderLock(Document doc) {
    String id = doc.id();
    Generation gen = this.generation;
    // Check id is not in live generation (and not tombstoned) and not in staging
    int liveOrd = gen.idMapper.contains(id) ? gen.idMapper.ordinalOf(id) : -1;
    boolean liveAndNotTombstoned = liveOrd >= 0 && !gen.tombstones.get(liveOrd);
    if (liveAndNotTombstoned || staging.contains(id)) {
      throw new IllegalArgumentException("Duplicate id: " + id);
    }
    staging.append(doc);
  }

  private void maybeAutoCommit() {
    if (staging.size() >= config.autoCommitThreshold()) {
      commitUnderLock();
    }
  }

  // ---------------------------------------------------------------------------
  // Delete / Upsert
  // ---------------------------------------------------------------------------

  @Override
  public boolean delete(String id) {
    Objects.requireNonNull(id, "id must not be null");
    writerLock.lock();
    try {
      ensureOpenUnderLock();
      return deleteUnderLock(id);
    } finally {
      writerLock.unlock();
    }
  }

  private boolean deleteUnderLock(String id) {
    // Case 1: id is in staging buffer (not-yet-committed add) — remove from staging.
    if (staging.contains(id)) {
      staging.removeDocument(id);
      return true;
    }
    // Case 2: id is in live generation AND not tombstoned AND not already pending tombstone.
    Generation gen = this.generation;
    int ord = gen.idMapper.ordinalOf(id);
    if (ord >= 0 && !gen.tombstones.get(ord) && !staging.isTombstoned(id)) {
      staging.stageDelete(id);
      return true;
    }
    // Case 3: unknown or already deleted.
    return false;
  }

  @Override
  public int deleteWhere(Filter filter) {
    Objects.requireNonNull(filter, "filter must not be null");
    writerLock.lock();
    try {
      ensureOpenUnderLock();
      int count = 0;
      Generation gen = this.generation;

      // Walk live generation ordinals, skip tombstoned and already-pending.
      for (int i = 0; i < gen.physicalCount; i++) {
        if (gen.tombstones.get(i)) {
          continue;
        }
        String id = gen.idMapper.idOf(i);
        if (staging.isTombstoned(id)) {
          continue;
        }
        Document doc = gen.metadataStore.get(i);
        if (doc != null && FilterExecutor.matches(filter, doc.metadata())) {
          staging.stageDelete(id);
          count++;
        }
      }

      // Also walk staged (not-yet-committed) documents and remove matches.
      List<String> stagedToRemove = new ArrayList<>();
      for (Document doc : staging.documents()) {
        if (FilterExecutor.matches(filter, doc.metadata())) {
          stagedToRemove.add(doc.id());
        }
      }
      for (String id : stagedToRemove) {
        staging.removeDocument(id);
        count++;
      }

      return count;
    } finally {
      writerLock.unlock();
    }
  }

  @Override
  public void upsert(Document doc) {
    validateForInsert(doc);
    writerLock.lock();
    try {
      ensureOpenUnderLock();
      String id = doc.id();

      // If id in staging buffer: remove old staged version.
      if (staging.contains(id)) {
        staging.removeDocument(id);
      } else {
        // If id in live generation and not tombstoned: stage tombstone for old ordinal.
        Generation gen = this.generation;
        int ord = gen.idMapper.ordinalOf(id);
        if (ord >= 0 && !gen.tombstones.get(ord) && !staging.isTombstoned(id)) {
          staging.stageDelete(id);
        }
      }

      // Stage new document (gets a new ordinal at the end).
      staging.append(doc);
      maybeAutoCommit();
    } finally {
      writerLock.unlock();
    }
  }

  // ---------------------------------------------------------------------------
  // Commit
  // ---------------------------------------------------------------------------

  @Override
  public void commit() {
    writerLock.lock();
    try {
      ensureOpenUnderLock();
      commitUnderLock();
      // Invalidate cached query results so subsequent searches reflect the new generation.
      queryCache.invalidateAll();
    } finally {
      writerLock.unlock();
    }
  }

  private void commitUnderLock() {
    if (!staging.hasWork()) {
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
  // In-memory commit
  // ---------------------------------------------------------------------------

  private void commitInMemory(Generation oldGen) {
    int oldPhysicalCount = oldGen.physicalCount;
    int stagedCount = staging.size();
    int newPhysicalCount = oldPhysicalCount + stagedCount;

    // Copy the old generation's tombstones and apply pending ones.
    BitSet newTombstones = (BitSet) oldGen.tombstones.clone();
    for (String id : staging.pendingTombstones()) {
      int ord = oldGen.idMapper.ordinalOf(id);
      if (ord >= 0) {
        newTombstones.set(ord);
      }
    }

    InMemoryIdMapper newMapper = InMemoryIdMapper.copyOf((InMemoryIdMapper) oldGen.idMapper);
    InMemoryMetadataStore newMeta =
        InMemoryMetadataStore.copyOf((InMemoryMetadataStore) oldGen.metadataStore);

    // Build the successor vector matrix. All ordinals — including tombstoned ones — retain their
    // original vectors so graph traversal never operates on zero-vector placeholders, which would
    // produce NaN cosine-similarity scores and corrupt the HNSW/Vamana priority-queue ordering.
    // Tombstoned ordinals are excluded from search results by the tombstone check in search(),
    // not by zeroing out their vector content. This makes in-memory mode consistent with the
    // persistent path, which bulk-copies vectors.bin byte-for-byte (tombstoned bytes included).
    //
    // Vector arrays are shared by reference with the predecessor generation, matching the
    // documented contract on {@link Document} ("the collection stores the reference directly for
    // zero-copy ingestion, and the same reference is handed back through the metadata store and
    // through the backend's vector matrix"). This makes commit O(newPhysicalCount) reference
    // copies instead of O(newPhysicalCount × dim) deep clones — a 100×+ reduction in commit-path
    // allocation for typical embedding dimensions. Without this, a write-heavy workload makes the
    // young-gen GC churn dominate concurrent reader latency.
    float[][] next = new float[newPhysicalCount][];
    for (int i = 0; i < oldPhysicalCount; i++) {
      Document stored = oldGen.metadataStore.get(i);
      if (stored == null) {
        throw new IllegalStateException(
            "Missing document in metadata store for ordinal " + i + " during commit");
      }
      next[i] = stored.vector();
    }
    List<Document> stagedDocs = staging.documents();
    for (int i = 0; i < stagedCount; i++) {
      Document doc = stagedDocs.get(i);
      // Use putOrReplace because an upserted doc may have the same id as a tombstoned ordinal
      // in the predecessor generation (the old forward mapping must be overwritten).
      int ordinal = newMapper.putOrReplace(doc.id());
      int expected = oldPhysicalCount + i;
      if (ordinal != expected) {
        throw new IllegalStateException(
            "Ordinal mismatch: expected " + expected + " but got " + ordinal);
      }
      newMeta.put(ordinal, doc);
      next[ordinal] = doc.vector();
    }

    IndexSpi newSpi = newInMemoryAdapter();
    newSpi.build(next, config.metric());

    // Train quantizer if configured.
    if (config.quantizerKind() != QuantizerKind.NONE
        && liveCountFrom(newPhysicalCount, newTombstones) > 0) {
      // Build a dataset of only live vectors for quantizer training.
      float[][] liveVectors = extractLiveVectors(next, newTombstones);
      if (liveVectors.length > 0) {
        VectorDataset dataset = new ArrayVectorDataset(liveVectors);
        TrainedQuantization tq = trainQuantizer(dataset, config);
        if (newSpi instanceof FlatScanAdapter fa) {
          newSpi = new QuantizedFlatScanAdapter(fa, fa, config.metric(), tq.compressed());
        } else if (newSpi instanceof HnswIndexAdapter ha) {
          ha.enableQuantization(tq.compressed());
        } else if (newSpi instanceof VamanaIndexAdapter va) {
          va.enableQuantization(tq.compressed());
        }
      }
    }

    Generation newGen =
        new Generation(
            newSpi, newMapper, newMeta, newPhysicalCount, newTombstones, 0L, null, null, null);
    this.generation = newGen;
    staging.clear();
    oldGen.release();
  }

  // ---------------------------------------------------------------------------
  // Persistent commit
  // ---------------------------------------------------------------------------

  private void commitPersistent(Generation oldGen) {
    int oldPhysicalCount = oldGen.physicalCount;
    int stagedCount = staging.size();
    int newPhysicalCount = oldPhysicalCount + stagedCount;
    int dim = config.dimension();

    // Apply tombstones.
    BitSet newTombstones = (BitSet) oldGen.tombstones.clone();
    for (String id : staging.pendingTombstones()) {
      int ord = oldGen.idMapper.ordinalOf(id);
      if (ord >= 0) {
        newTombstones.set(ord);
      }
    }

    // 1. Build the ordered (id, document) lists for the successor generation.
    //    ALL ordinals 0..physicalCount-1 are included (including tombstoned).
    List<String> newIds = new ArrayList<>(newPhysicalCount);
    List<Document> newDocs = new ArrayList<>(newPhysicalCount);
    for (int i = 0; i < oldPhysicalCount; i++) {
      newIds.add(oldGen.idMapper.idOf(i));
      newDocs.add(oldGen.metadataStore.get(i));
    }
    for (Document d : staging.documents()) {
      newIds.add(d.id());
      newDocs.add(d);
    }

    // 2. Build the data-file byte images.
    boolean needMatrix =
        config.indexType() == IndexType.HNSW
            || config.indexType() == IndexType.VAMANA
            || config.indexType() == IndexType.FLAT && config.quantizerKind() != QuantizerKind.NONE
            || config.indexType() == IndexType.IVF_FLAT
            || config.indexType() == IndexType.IVF_PQ;
    Materialized materialized =
        materializeSuccessor(
            oldGen.mappedVectors, oldPhysicalCount, staging.documents(), dim, needMatrix);
    byte[] vectorsBin = materialized.vectorsBin();
    byte[] idmapBin;
    byte[] metadataBin;
    try {
      idmapBin = MappedIdMapper.Writer.toBytes(newIds);
      metadataBin = MappedMetadataStore.Writer.toBytes(newDocs);
    } catch (IOException e) {
      throw new UncheckedIOException("Failed to serialize commit payload", e);
    }

    // 2a. Graph bytes.
    byte[] graphBin = null;
    long graphBinLength = 0L;
    long graphBinCrc = 0L;
    boolean needGraph =
        config.indexType() == IndexType.HNSW
            || config.indexType() == IndexType.VAMANA
            || config.indexType() == IndexType.IVF_FLAT
            || config.indexType() == IndexType.IVF_PQ;
    if (needGraph) {
      graphBin = encodeGraphBytes(materialized.matrix());
      if (graphBin != null) {
        graphBinLength = (long) graphBin.length;
        graphBinCrc = Checksums.ofBytes(graphBin);
      }
    }

    // 2b. Quantization.
    byte[] quantizedBin = null;
    long quantizedBinLength = 0L;
    long quantizedBinCrc = 0L;
    int liveCount = liveCountFrom(newPhysicalCount, newTombstones);
    if (config.quantizerKind() != QuantizerKind.NONE
        && liveCount > 0
        && materialized.matrix() != null) {
      float[][] liveVectors = extractLiveVectors(materialized.matrix(), newTombstones);
      if (liveVectors.length > 0) {
        VectorDataset dataset = new ArrayVectorDataset(liveVectors);
        TrainedQuantization trainedQ = trainQuantizer(dataset, config);
        quantizedBin =
            QuantizedVectorsCodec.encode(
                trainedQ.compressed(), trainedQ.quantizer(), config.quantizerKind());
        quantizedBinLength = (long) quantizedBin.length;
        quantizedBinCrc = Checksums.ofBytes(quantizedBin);
      }
    }

    // 2c. Tombstones.
    byte[] tombstonesBin = TombstoneCodec.encode(newTombstones, newPhysicalCount);
    long tombstonesBinLength = (long) tombstonesBin.length;
    long tombstonesBinCrc = tombstonesBin.length > 0 ? Checksums.ofBytes(tombstonesBin) : 0L;
    int tombstoneCount = newTombstones.cardinality();

    // 3. Build the manifest.
    long newGenNumber = nextGenerationNumber;
    Manifest manifest =
        Manifest.buildWithTombstones(
            config,
            newGenNumber,
            (long) liveCount,
            (long) vectorsBin.length,
            Checksums.ofBytes(vectorsBin),
            (long) metadataBin.length,
            Checksums.ofBytes(metadataBin),
            (long) idmapBin.length,
            Checksums.ofBytes(idmapBin),
            graphBinLength,
            graphBinCrc,
            quantizedBinLength,
            quantizedBinCrc,
            (long) tombstoneCount,
            tombstonesBinLength,
            tombstonesBinCrc);

    GenerationDirectory.WriteResult wr;
    try {
      wr =
          GenerationDirectory.writeGeneration(
              config.storageRoot(),
              newGenNumber,
              new BufferedGenerationSource(
                  vectorsBin, idmapBin, metadataBin, graphBin, quantizedBin, tombstonesBin),
              manifest);
    } catch (IOException e) {
      throw new UncheckedIOException("Failed to write generation " + newGenNumber, e);
    }

    nextGenerationNumber = newGenNumber + 1L;

    Generation newGen;
    try {
      newGen = openGeneration(wr.generationDir(), wr.manifest());
    } catch (IOException e) {
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

  // ---------------------------------------------------------------------------
  // Compact
  // ---------------------------------------------------------------------------

  @Override
  public void compact() {
    writerLock.lock();
    try {
      ensureOpenUnderLock();
      // Commit any pending work first.
      if (staging.hasWork()) {
        commitUnderLock();
      }
      Generation gen = this.generation;
      if (gen.tombstones.isEmpty()) {
        return; // no-op if no tombstones
      }
      if (config.storageRoot() == null) {
        compactInMemory(gen);
      } else {
        compactPersistent(gen);
      }
    } finally {
      writerLock.unlock();
    }
  }

  private void compactInMemory(Generation oldGen) {
    int liveCount = oldGen.liveCount();
    InMemoryIdMapper newMapper = new InMemoryIdMapper();
    InMemoryMetadataStore newMeta = new InMemoryMetadataStore();
    float[][] next = new float[liveCount][];

    // Build oldToNew mapping alongside the live-document gather loop so we can
    // pass it to HnswGraphMerger without a second pass.
    int[] oldToNew = new int[oldGen.physicalCount];
    Arrays.fill(oldToNew, -1);
    int slot = 0;
    for (int i = 0; i < oldGen.physicalCount; i++) {
      if (oldGen.tombstones.get(i)) {
        continue;
      }
      String id = oldGen.idMapper.idOf(i);
      Document doc = oldGen.metadataStore.get(i);
      if (doc == null) {
        continue;
      }
      oldToNew[i] = slot;
      int ordinal = newMapper.put(id);
      newMeta.put(ordinal, doc);
      next[slot] = doc.vector() != null ? doc.vector().clone() : new float[config.dimension()];
      slot++;
    }

    IndexSpi newSpi = newInMemoryAdapter();

    // IGTM: for HNSW, merge the existing graph instead of rebuilding from scratch.
    // The merger remaps surviving edges and repairs under-connected nodes via beam search —
    // O(N'·M·d + R·ef·M) vs O(N'·log N'·M·d) for a full rebuild.
    if (config.indexType() == IndexType.HNSW
        && liveCount > 0
        && newSpi instanceof HnswIndexAdapter ha
        && oldGen.spi instanceof HnswIndexAdapter oldHa
        && oldHa.graph() != null) {
      ha.mergeFrom(oldHa.graph(), next, oldToNew, config.metric());
    } else {
      newSpi.build(next, config.metric());
    }

    // Retrain quantizer on compacted data.
    if (config.quantizerKind() != QuantizerKind.NONE && liveCount > 0) {
      VectorDataset dataset = new ArrayVectorDataset(next);
      TrainedQuantization tq = trainQuantizer(dataset, config);
      if (newSpi instanceof FlatScanAdapter fa) {
        newSpi = new QuantizedFlatScanAdapter(fa, fa, config.metric(), tq.compressed());
      } else if (newSpi instanceof HnswIndexAdapter ha) {
        ha.enableQuantization(tq.compressed());
      } else if (newSpi instanceof VamanaIndexAdapter va) {
        va.enableQuantization(tq.compressed());
      }
    }

    Generation newGen =
        new Generation(newSpi, newMapper, newMeta, liveCount, new BitSet(), 0L, null, null, null);
    this.generation = newGen;
    oldGen.release();
  }

  private void compactPersistent(Generation oldGen) {
    // Gather only live documents into dense ordinal order.
    // Build oldToNew in the same loop for the IGTM HNSW merge path.
    int liveCount = oldGen.liveCount();
    List<String> newIds = new ArrayList<>(liveCount);
    List<Document> newDocs = new ArrayList<>(liveCount);
    float[][] matrix = new float[liveCount][];
    int[] oldToNew = new int[oldGen.physicalCount];
    Arrays.fill(oldToNew, -1);

    int slot = 0;
    for (int i = 0; i < oldGen.physicalCount; i++) {
      if (oldGen.tombstones.get(i)) {
        continue;
      }
      String id = oldGen.idMapper.idOf(i);
      Document doc = oldGen.metadataStore.get(i);
      newIds.add(id);
      newDocs.add(doc);
      oldToNew[i] = slot;
      // Hydrate vector from mmap if needed.
      if (doc != null && doc.vector() != null) {
        matrix[slot] = doc.vector().clone();
      } else if (oldGen.mappedVectors != null) {
        float[] v = new float[config.dimension()];
        MemorySegment.copy(
            oldGen.mappedVectors.vectorSlice(i),
            ValueLayout.JAVA_FLOAT,
            0L,
            v,
            0,
            config.dimension());
        matrix[slot] = v;
      } else {
        matrix[slot] = new float[config.dimension()];
      }
      slot++;
    }

    int dim = config.dimension();

    // Build vectors.bin from the compacted matrix.
    byte[] vectorsBin = buildVectorsBinFromMatrix(matrix, dim);
    byte[] idmapBin;
    byte[] metadataBin;
    try {
      idmapBin = MappedIdMapper.Writer.toBytes(newIds);
      metadataBin = MappedMetadataStore.Writer.toBytes(newDocs);
    } catch (IOException e) {
      throw new UncheckedIOException("Failed to serialize compact payload", e);
    }

    // Graph.
    byte[] graphBin = null;
    long graphBinLength = 0L;
    long graphBinCrc = 0L;
    boolean needGraph =
        config.indexType() == IndexType.HNSW
            || config.indexType() == IndexType.VAMANA
            || config.indexType() == IndexType.IVF_FLAT
            || config.indexType() == IndexType.IVF_PQ;
    if (needGraph && liveCount > 0) {
      // IGTM: for HNSW, merge the existing graph instead of rebuilding from scratch.
      if (config.indexType() == IndexType.HNSW
          && oldGen.spi instanceof MappedHnswIndexAdapter mapped) {
        HnswGraph oldGraph = mapped.graph();
        if (oldGraph != null) {
          VectorCollectionConfig.HnswParams hp = config.hnswParams();
          HnswGraph merged =
              HnswGraphMerger.merge(
                  oldGraph, matrix, oldToNew, config.metric(), hp.m(), hp.efConstruction());
          graphBin = merged != null ? HnswGraphCodec.encode(merged) : null;
        } else {
          graphBin = encodeGraphBytes(matrix);
        }
      } else {
        graphBin = encodeGraphBytes(matrix);
      }
      if (graphBin != null) {
        graphBinLength = (long) graphBin.length;
        graphBinCrc = Checksums.ofBytes(graphBin);
      }
    }

    // Quantization.
    byte[] quantizedBin = null;
    long quantizedBinLength = 0L;
    long quantizedBinCrc = 0L;
    TrainedQuantization trainedQ = null;
    if (config.quantizerKind() != QuantizerKind.NONE && liveCount > 0) {
      VectorDataset dataset = new ArrayVectorDataset(matrix);
      trainedQ = trainQuantizer(dataset, config);
      quantizedBin =
          QuantizedVectorsCodec.encode(
              trainedQ.compressed(), trainedQ.quantizer(), config.quantizerKind());
      quantizedBinLength = (long) quantizedBin.length;
      quantizedBinCrc = Checksums.ofBytes(quantizedBin);
    }

    // No tombstones after compact.
    long newGenNumber = nextGenerationNumber;
    Manifest manifest =
        Manifest.buildWithTombstones(
            config,
            newGenNumber,
            (long) liveCount,
            (long) vectorsBin.length,
            Checksums.ofBytes(vectorsBin),
            (long) metadataBin.length,
            Checksums.ofBytes(metadataBin),
            (long) idmapBin.length,
            Checksums.ofBytes(idmapBin),
            graphBinLength,
            graphBinCrc,
            quantizedBinLength,
            quantizedBinCrc,
            0L,
            0L,
            0L);

    GenerationDirectory.WriteResult wr;
    try {
      wr =
          GenerationDirectory.writeGeneration(
              config.storageRoot(),
              newGenNumber,
              new BufferedGenerationSource(
                  vectorsBin, idmapBin, metadataBin, graphBin, quantizedBin, null),
              manifest);
    } catch (IOException e) {
      throw new UncheckedIOException("Failed to write compacted generation " + newGenNumber, e);
    }

    nextGenerationNumber = newGenNumber + 1L;

    Generation newGen;
    try {
      newGen = openGeneration(wr.generationDir(), wr.manifest());
    } catch (IOException e) {
      throw new UncheckedIOException("Compacted generation cannot be opened", e);
    }

    this.generation = newGen;
    oldGen.release();
  }

  /** Builds a vectors.bin byte image from a float[][] matrix. */
  private static byte[] buildVectorsBinFromMatrix(float[][] matrix, int dim) {
    long strideL = AlignmentUtil.alignUp((long) dim * Float.BYTES, AlignmentUtil.VECTOR_ALIGNMENT);
    int stride = (int) strideL;
    long totalL = strideL * (long) matrix.length;
    if (totalL > Integer.MAX_VALUE) {
      throw new IllegalStateException("vectors.bin exceeds 2 GiB: " + totalL);
    }
    byte[] out = new byte[(int) totalL];
    int rawVecBytes = dim * Float.BYTES;
    ByteBuffer buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
    for (float[] v : matrix) {
      for (int j = 0; j < dim; j++) {
        buf.putFloat(v[j]);
      }
      buf.position(buf.position() + (stride - rawVecBytes));
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // Materialization helpers
  // ---------------------------------------------------------------------------

  private record Materialized(byte[] vectorsBin, float[][] matrix) {}

  private static Materialized materializeSuccessor(
      MemorySegmentVectors oldMapped,
      int physicalCount,
      List<Document> staged,
      int dim,
      boolean needMatrix) {
    long strideL = AlignmentUtil.alignUp((long) dim * Float.BYTES, AlignmentUtil.VECTOR_ALIGNMENT);
    if (strideL > Integer.MAX_VALUE) {
      throw new IllegalStateException("vector stride exceeds 2 GiB: " + strideL);
    }
    int stride = (int) strideL;
    int newSize = physicalCount + staged.size();
    long totalL = strideL * (long) newSize;
    if (totalL > Integer.MAX_VALUE) {
      throw new IllegalStateException("vectors.bin exceeds 2 GiB: " + totalL);
    }
    byte[] out = new byte[(int) totalL];
    float[][] matrix = needMatrix ? new float[newSize][] : null;

    // Bulk-copy the old generation byte-for-byte.
    if (physicalCount > 0 && oldMapped != null) {
      long oldBytes = strideL * physicalCount;
      MemorySegment.copy(oldMapped.segment(), ValueLayout.JAVA_BYTE, 0L, out, 0, (int) oldBytes);
      if (needMatrix) {
        for (int i = 0; i < physicalCount; i++) {
          float[] v = new float[dim];
          MemorySegment.copy(oldMapped.vectorSlice(i), ValueLayout.JAVA_FLOAT, 0L, v, 0, dim);
          matrix[i] = v;
        }
      }
    }

    int rawVecBytes = dim * Float.BYTES;
    ByteBuffer buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
    buf.position(stride * physicalCount);
    for (int s = 0; s < staged.size(); s++) {
      float[] v = staged.get(s).vector();
      if (needMatrix) {
        matrix[physicalCount + s] = v.clone();
      }
      for (int j = 0; j < dim; j++) {
        buf.putFloat(v[j]);
      }
      buf.position(buf.position() + (stride - rawVecBytes));
    }
    return new Materialized(out, matrix);
  }

  private byte[] encodeGraphBytes(float[][] matrix) {
    return switch (config.indexType()) {
      case HNSW -> {
        VectorCollectionConfig.HnswParams hp = config.hnswParams();
        HnswIndexAdapter adapter = new HnswIndexAdapter(hp.m(), hp.efConstruction(), hp.threads());
        adapter.build(matrix, config.metric());
        HnswGraph graph = adapter.graph();
        yield graph == null ? null : HnswGraphCodec.encode(graph);
      }
      case VAMANA -> {
        VectorCollectionConfig.VamanaParams vp = config.vamanaParams();
        VamanaIndexAdapter adapter =
            new VamanaIndexAdapter(
                vp.maxDegree(), vp.searchListSize(), vp.alpha(), vp.seed(), vp.threads());
        adapter.build(matrix, config.metric());
        VamanaGraph graph = adapter.graph();
        yield graph == null ? null : VamanaGraphCodec.encode(graph);
      }
      case IVF_FLAT -> {
        if (matrix == null || matrix.length == 0) {
          yield null;
        }
        VectorCollectionConfig.IvfParams p = config.ivfParams();
        int effectiveK = Math.min(p.k(), matrix.length);
        IvfBuildParams bp = new IvfBuildParams(effectiveK, p.maxIter(), 0f, p.soar(), p.seed(), 0);
        IvfIndex idx = IvfIndex.build(matrix, null, config.metric(), bp);
        yield idx.encode();
      }
      case IVF_PQ -> {
        if (matrix == null || matrix.length == 0) {
          yield null;
        }
        VectorCollectionConfig.IvfPqParams p = config.ivfPqParams();
        int effectiveK = Math.min(p.k(), matrix.length);
        IvfBuildParams base =
            new IvfBuildParams(effectiveK, p.maxIter(), 0f, p.soar(), p.seed(), 0);
        IvfBuildParams bp =
            base.withPq(p.pqSubspaces(), p.pqClusters(), p.pqAnisotropicThreshold());
        IvfIndex idx = IvfIndex.build(matrix, null, config.metric(), bp);
        yield idx.encode();
      }
      case FLAT ->
          throw new IllegalStateException(
              "encodeGraphBytes called with non-graph indexType " + config.indexType());
      case CUVS_BRUTEFORCE, CUVS_CAGRA ->
          throw new UnsupportedOperationException(
              "CUVS_* index types do not support persistent storage yet: " + config.indexType());
    };
  }

  // ---------------------------------------------------------------------------
  // Quantization training
  // ---------------------------------------------------------------------------

  private record TrainedQuantization(Quantizer<?> quantizer, CompressedVectors compressed) {}

  @SuppressWarnings("unchecked")
  private static TrainedQuantization trainQuantizer(
      VectorDataset dataset, VectorCollectionConfig config) {
    QuantizerParams params = config.quantizerParams();
    Quantizer<?> quantizer =
        switch (config.quantizerKind()) {
          case NONE -> throw new IllegalStateException("trainQuantizer called with NONE");
          case SQ8 -> ScalarQuantizer.train(dataset, ScalarBits.INT8);
          case SQ4 -> ScalarQuantizer.train(dataset, ScalarBits.INT4);
          case PQ -> {
            QuantizerParams.PqParams pq =
                params instanceof QuantizerParams.PqParams p
                    ? p
                    : new QuantizerParams.PqParams(
                        Math.max(1, config.dimension() / 8),
                        VectorCollectionBuilder.DEFAULT_PQ_CLUSTERS,
                        true);
            yield ProductQuantizer.train(
                dataset, pq.numSubspaces(), pq.numClusters(), pq.center(), pq.trainThreads());
          }
          case BQ -> {
            boolean bbq = params instanceof QuantizerParams.BqParams b ? b.bbq() : true;
            yield BinaryQuantizer.train(dataset, bbq ? BinaryMode.BBQ : BinaryMode.SIGN_BIT);
          }
          case RABITQ -> {
            long seed =
                params instanceof QuantizerParams.RaBitParams r
                    ? r.seed()
                    : VectorCollectionBuilder.DEFAULT_RABIT_SEED;
            yield RaBitQuantizer.train(dataset, seed);
          }
          case NVQ -> {
            int numSv =
                params instanceof QuantizerParams.NvqParams n
                    ? n.numSubvectors()
                    : Math.max(1, config.dimension() / 4);
            yield NVQuantizer.train(dataset, numSv);
          }
        };
    @SuppressWarnings("rawtypes")
    CompressedVectors compressed = ((Quantizer) quantizer).encodeAll(dataset);
    return new TrainedQuantization(quantizer, compressed);
  }

  // ---------------------------------------------------------------------------
  // Tombstone helpers
  // ---------------------------------------------------------------------------

  private static int liveCountFrom(int physicalCount, BitSet tombstones) {
    return physicalCount - tombstones.cardinality();
  }

  /**
   * Extracts only the live (non-tombstoned) vectors from a matrix. Used for quantizer training
   * which should not see zero-placeholder vectors.
   */
  private static float[][] extractLiveVectors(float[][] matrix, BitSet tombstones) {
    int liveCount = matrix.length - tombstones.cardinality();
    float[][] live = new float[liveCount][];
    int slot = 0;
    for (int i = 0; i < matrix.length; i++) {
      if (!tombstones.get(i)) {
        live[slot++] = matrix[i];
      }
    }
    return live;
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
    boolean hasFilter = filter != null && !(filter instanceof Filter.All);

    // QvCache: only cache results that do not embed vectors (ids + scores + text + metadata are
    // stable across snapshots; vectors require mmap reads that callers may store separately).
    boolean cacheEligible = !request.includeVector();
    if (cacheEligible) {
      var cached = queryCache.get(request.query(), request.k(), filter);
      if (cached.isPresent()) return cached.get();
    }

    Generation gen = acquireReadSnapshot();
    try {
      long start = System.nanoTime();

      // ACORN pre-filter: HNSW SPI implementations navigate non-matching nodes for routing
      // but only collect matching ones in the result set — avoids recall collapse on selective
      // filters without expanding the candidate pool.
      boolean supportsPreFilter =
          gen.spi instanceof HnswIndexAdapter || gen.spi instanceof MappedHnswIndexAdapter;
      boolean preFilterActive = hasFilter && supportsPreFilter;

      int candidateK = request.k();
      int candidateSearchListSize = request.searchListSize();
      IndexSpi.SearchOutcome outcome;

      if (preFilterActive) {
        // Build an ordinal-level predicate combining tombstone + metadata filter.
        final Generation snapshot = gen;
        final Filter f = filter;
        IntPredicate pred =
            ordinal -> {
              if (snapshot.tombstones.get(ordinal)) return false;
              Document doc = snapshot.metadataStore.get(ordinal);
              if (doc == null) return false;
              return FilterExecutor.matches(f, doc.metadata());
            };
        // No candidate expansion needed — the predicate enforces filter during traversal.
        // Phase 1: filtered multi-start is deferred; ignore searchMultiStart here.
        if (request.searchMultiStart() > 1) {
          LOGGER.log(
              java.util.logging.Level.FINE,
              "searchMultiStart={0} ignored on ACORN pre-filter path (Phase 1 scope)",
              request.searchMultiStart());
        }
        outcome =
            gen.spi.searchWithPredicate(
                request.query(),
                candidateK,
                candidateSearchListSize,
                request.overQueryFactor(),
                pred);
      } else {
        // Post-filter path: expand the candidate pool so the filter has enough to choose from.
        if (hasFilter) {
          candidateK =
              Math.min(
                  Math.max((int) Math.ceil(request.k() * request.filterExpansion()), request.k()),
                  Math.max(gen.liveCount(), request.k()));
          candidateSearchListSize = Math.max(candidateK, request.searchListSize());
        }
        if (request.searchMultiStart() > 1) {
          outcome =
              gen.spi.search(
                  request.query(),
                  candidateK,
                  candidateSearchListSize,
                  request.overQueryFactor(),
                  request.searchMultiStart());
        } else {
          outcome =
              gen.spi.search(
                  request.query(), candidateK, candidateSearchListSize, request.overQueryFactor());
        }
      }

      int[] ordinals = outcome.ordinals();
      float[] scores = outcome.scores();

      List<SearchResult.Hit> hits = new ArrayList<>(Math.min(ordinals.length, request.k()));
      for (int i = 0; i < ordinals.length && hits.size() < request.k(); i++) {
        int ordinal = ordinals[i];
        // Tombstone + metadata filter: skip when ACORN predicate already handled them.
        if (!preFilterActive) {
          if (gen.tombstones.get(ordinal)) {
            continue;
          }
        }
        float score = scores[i];
        if (score < request.minScore()) {
          continue;
        }
        Document stored = gen.metadataStore.get(ordinal);
        if (stored == null) {
          continue;
        }
        if (!preFilterActive && hasFilter && !FilterExecutor.matches(filter, stored.metadata())) {
          continue;
        }
        float[] vector = null;
        if (request.includeVector()) {
          vector = stored.vector();
          if (vector == null && gen.mappedVectors != null) {
            vector = new float[config.dimension()];
            MemorySegment.copy(
                gen.mappedVectors.vectorSlice(ordinal),
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

      if (hasFilter && hits.size() < request.k() && gen.liveCount() >= request.k()) {
        LOGGER.log(
            Level.FINE,
            "Filtered search returned {0} of {1} requested results "
                + "(filter was too selective for the candidate pool of {2})",
            new Object[] {hits.size(), request.k(), ordinals.length});
      }

      long elapsed = System.nanoTime() - start;
      SearchResult result = new SearchResult(hits, elapsed);
      if (cacheEligible) {
        queryCache.put(request.query(), request.k(), filter, result);
      }
      return result;
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
      if (ord < 0 || gen.tombstones.get(ord)) {
        return null;
      }
      return gen.metadataStore.get(ord);
    } finally {
      gen.release();
    }
  }

  @Override
  public boolean contains(String id) {
    Objects.requireNonNull(id, "id must not be null");
    Generation gen = acquireReadSnapshot();
    try {
      int ord = gen.idMapper.ordinalOf(id);
      return ord >= 0 && !gen.tombstones.get(ord);
    } finally {
      gen.release();
    }
  }

  @Override
  public List<Document> documents() {
    Generation gen = acquireReadSnapshot();
    try {
      List<Document> result = new ArrayList<>(gen.liveCount());
      for (int i = 0; i < gen.physicalCount; i++) {
        if (gen.tombstones.get(i)) continue;
        Document doc = gen.metadataStore.get(i);
        if (doc == null) continue;
        // Hydrate vector from mmap if not embedded in the document (persistent generations).
        if (doc.vector() == null && gen.mappedVectors != null) {
          float[] v = new float[config.dimension()];
          MemorySegment.copy(
              gen.mappedVectors.vectorSlice(i),
              ValueLayout.JAVA_FLOAT,
              0L,
              v,
              0,
              config.dimension());
          doc = new Document(doc.id(), v, doc.text(), doc.metadata());
        }
        result.add(doc);
      }
      return Collections.unmodifiableList(result);
    } finally {
      gen.release();
    }
  }

  @Override
  public int size() {
    Generation gen = acquireReadSnapshot();
    try {
      return gen.liveCount();
    } finally {
      gen.release();
    }
  }

  @Override
  public int physicalSize() {
    Generation gen = acquireReadSnapshot();
    try {
      return gen.physicalCount;
    } finally {
      gen.release();
    }
  }

  @Override
  public VectorCollectionConfig config() {
    return config;
  }

  @Override
  public void flush() {
    // No-op in both modes.
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
      gen.release();
    } finally {
      writerLock.unlock();
    }
  }

  private Generation acquireReadSnapshot() {
    while (true) {
      Generation gen = this.generation;
      if (gen == null) {
        throw new IllegalStateException("VectorCollection is closed");
      }
      if (gen.acquire()) {
        return gen;
      }
    }
  }

  private void ensureOpenUnderLock() {
    if (this.generation == null) {
      throw new IllegalStateException("VectorCollection is closed");
    }
  }
}
