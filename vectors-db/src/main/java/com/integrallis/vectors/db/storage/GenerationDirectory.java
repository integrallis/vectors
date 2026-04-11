package com.integrallis.vectors.db.storage;

import com.integrallis.vectors.db.IndexType;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.AccessDeniedException;
import java.nio.file.DirectoryStream;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Orchestrates the on-disk write protocol for a single {@code vectors-db} collection: writing a new
 * generation into an atomically-published {@code gen-NNNNNNNNNNNNNNNN/} directory, updating the
 * collection-root {@code CURRENT} pointer, and running the crash-recovery sweep on open.
 *
 * <p>This class is a stateless namespace — every method is {@code static}. The rationale is the
 * same as {@link Checksums} and {@link FileFormat}: the generation-directory protocol is a pure
 * function of its inputs (root path, generation number, source callbacks) and holds no per-instance
 * state worth caching. Making it a namespace keeps test setup dead simple.
 *
 * <p><b>Write protocol.</b> {@link #writeGeneration(Path, long, GenerationSource, Manifest)} runs
 * the full crash-safe commit pipeline documented in the Step 4a design doc §"Commit sequence
 * (persistent mode)":
 *
 * <ol>
 *   <li>Create the in-flight tmp dir {@code .gen-NNNNNNNNNNNNNNNN.tmp/} under the collection root.
 *   <li>Invoke {@link GenerationSource} callbacks to materialize {@code vectors.bin}, {@code
 *       idmap.bin}, and {@code metadata.bin} inside the tmp dir — plus {@code graph.bin} if the
 *       manifest declares a graph index (currently {@link
 *       com.integrallis.vectors.db.IndexType#HNSW}). Each callback is expected to {@link
 *       FileChannel#force(boolean) fsync} its own file contents — the tmp dir directory entries are
 *       fsynced by this method.
 *   <li>Write {@code manifest.bin} via {@link Manifest#writeTo(Path)} (which fsyncs the file).
 *   <li>Fsync the tmp dir so every directory entry is durable.
 *   <li>Atomic rename {@code .gen-NNNN.tmp/ → gen-NNNN/}.
 *   <li>Fsync the collection root so the rename itself is durable.
 *   <li>Atomically publish {@code CURRENT} via {@link #writeCurrentAtomic(Path, long)} (write
 *       {@code CURRENT.tmp}, fsync, rename, fsync parent).
 * </ol>
 *
 * <p>A crash at any step before {@code CURRENT} is updated leaves the collection in a recoverable
 * state that {@link #recover(Path, GenerationSource, Manifest)} will heal on the next open — see
 * the recovery semantics below.
 *
 * <p><b>Recovery protocol.</b> {@link #recover(Path, GenerationSource, Manifest)} is the inverse
 * operation invoked by the facade on every {@code open()}. Given a storage root, it:
 *
 * <ol>
 *   <li>Deletes every {@code .gen-*.tmp/} tmp directory (partial writes from a crashed commit).
 *   <li>Deletes {@code CURRENT.tmp} if present (partial CURRENT write).
 *   <li>If {@code CURRENT} exists and points at a {@code gen-NNNN/} whose manifest validates AND
 *       every payload file ({@code vectors.bin}, {@code idmap.bin}, {@code metadata.bin}, {@code
 *       graph.bin}) matches the per-file CRC stored in the manifest, opens that generation.
 *   <li>If {@code CURRENT} is missing, or its target has a corrupt manifest, or its target has any
 *       corrupt payload file, scans for the newest {@code gen-NNNN/} (by generation number,
 *       descending) whose manifest AND every payload file validates, and rewrites {@code CURRENT}
 *       to point at it via {@link #writeCurrentAtomic(Path, long)}. A manifest that parses cleanly
 *       but references a corrupt {@code vectors.bin} or {@code graph.bin} is treated identically to
 *       a corrupt manifest — the walk-back lets operators heal bit-rot or a partially-truncated
 *       write by rolling forward to an intact older generation without manual intervention.
 *   <li>If no valid generation exists at all, initializes an empty {@code gen-0000000000000000/}
 *       from the provided {@link GenerationSource} and publishes it. This is the "fresh open on
 *       empty directory" path.
 * </ol>
 *
 * <p>The recovery sweep is a pure function of directory state: running it twice in a row is
 * idempotent because run #1 leaves the directory in a shape where run #2 will find {@code CURRENT}
 * already consistent and do nothing.
 *
 * <p><b>CURRENT is authoritative.</b> If {@code CURRENT} points at a valid generation, that
 * generation is the live one — even if a newer numerically-greater {@code gen-NNNN/} exists on
 * disk. The scan-fallback branch only fires when {@code CURRENT} itself is missing or its target is
 * corrupt. Consequently, an "orphan" generation directory left behind by a process that wrote
 * {@code gen-N+1/} but crashed before updating {@code CURRENT} will <b>not</b> be promoted on the
 * next open — the live state will roll back to whatever {@code CURRENT} still references. This is
 * deliberate: the alternative (always taking the newest gen on disk) would silently undo a {@code
 * compact()} that retired old generations but kept their directories around for a reader-grace
 * period. The cost is that an orphaned commit is treated as a no-op rather than a recovered write;
 * in practice writers only ever stage one in-flight commit at a time, so the only path that can
 * produce orphans is a crash strictly between the {@code gen-NNNN/} rename and the {@code
 * CURRENT.tmp} write — a window of a few microseconds.
 *
 * <p><b>Empty-directory bootstrap.</b> Opening a brand-new empty path creates {@code
 * gen-0000000000000000/} with an empty payload. Callers must supply a {@link GenerationSource} that
 * knows how to produce an empty-but-valid set of files — {@code MappedIdMapper.Writer.writeTo(...,
 * List.of())} plus {@code MappedMetadataStore.Writer.writeTo(..., List.of())} plus an empty {@code
 * vectors.bin} file — and the corresponding empty {@link Manifest}. The sole branch that requires
 * this is case 5 of the recovery protocol; callers that don't want bootstrap semantics can pass
 * {@code null} and the method will instead throw {@link IOException} on an empty root.
 *
 * <p><b>File alignment.</b> This class does not know or care how {@code vectors.bin}, {@code
 * idmap.bin}, or {@code metadata.bin} are laid out. It only ensures that (a) they're written before
 * the manifest, (b) every file is fsynced before the tmp rename, and (c) the tmp rename precedes
 * the {@code CURRENT} update. The {@code GenerationSource} callbacks carry all of the file-format
 * knowledge; see {@link MappedIdMapper.Writer}, {@link MappedMetadataStore.Writer}, and {@link
 * com.integrallis.vectors.storage.store.VectorStoreWriter} for the writer helpers.
 */
public final class GenerationDirectory {

  private static final Logger LOGGER = Logger.getLogger(GenerationDirectory.class.getName());

  /**
   * Index types that own an on-disk {@code graph.bin} payload. A {@link
   * GenerationSource#writeGraph} callback is only invoked when {@code manifest.indexType()} is one
   * of these. Adding a new graph-backed index (e.g. persistent Vamana in Step 4c) means extending
   * this set and teaching the commit path to pass non-empty graph bytes through {@code
   * BufferedGenerationSource}.
   */
  private static final Set<IndexType> GRAPH_INDEX_TYPES = EnumSet.of(IndexType.HNSW);

  private GenerationDirectory() {}

  // ---------------------------------------------------------------------------
  // Source callback — supplied by the caller, materializes the three data files
  // into a directory that this class has already created. The manifest is not
  // part of this interface because this class already has a Manifest record in
  // hand and writes it directly via Manifest.writeTo().
  // ---------------------------------------------------------------------------

  /**
   * Strategy callback for writing a generation's data files into an in-flight tmp dir. Each
   * callback receives an absolute {@link Path} to the target file inside the tmp directory and is
   * responsible for writing the file contents and fsyncing the file data (for example via {@link
   * FileChannel#force(boolean) FileChannel.force(true)} or by closing a {@code VectorStoreWriter}
   * whose {@code ChannelOutput.flush()} calls {@link FileChannel#force(boolean) force(false)}). The
   * containing directory's own entries are fsynced later by {@link GenerationDirectory}.
   *
   * <p>The three core methods ({@code writeVectors}, {@code writeIdmap}, {@code writeMetadata}) are
   * invoked unconditionally in a single call to {@link GenerationDirectory#writeGeneration(Path,
   * long, GenerationSource, Manifest)} in the order {@code vectors → idmap → metadata → graph}. The
   * optional {@code writeGraph} callback is only invoked when the manifest's {@link
   * Manifest#indexType() indexType} is a graph index (currently {@link
   * com.integrallis.vectors.db.IndexType#HNSW}); for flat-scan generations it is skipped entirely
   * and no {@code graph.bin} file is written. If any callback throws, the tmp directory is cleaned
   * up before the exception propagates.
   *
   * <p>{@code writeGraph} defaults to a no-op so that existing flat-scan sources compile unchanged
   * after the Step 4b interface extension. A graph-backed source overrides it to emit the encoded
   * adjacency bytes (typically via {@code HnswGraphCodec.encode}).
   */
  public interface GenerationSource {
    /** Writes {@code vectors.bin} into the tmp dir at the given absolute path. */
    void writeVectors(Path destination) throws IOException;

    /** Writes {@code idmap.bin} into the tmp dir at the given absolute path. */
    void writeIdmap(Path destination) throws IOException;

    /** Writes {@code metadata.bin} into the tmp dir at the given absolute path. */
    void writeMetadata(Path destination) throws IOException;

    /**
     * Writes {@code graph.bin} into the tmp dir at the given absolute path. Invoked by {@link
     * GenerationDirectory#writeGeneration(Path, long, GenerationSource, Manifest)} iff the manifest
     * declares a graph index type. The default implementation is a no-op so flat-scan sources need
     * no override; graph-backed sources override to emit the encoded adjacency bytes.
     */
    default void writeGraph(Path destination) throws IOException {
      // Default: no graph file. Flat-scan sources keep this default;
      // graph sources override to emit graph.bin bytes.
    }
  }

  /**
   * Result of a {@link #writeGeneration(Path, long, GenerationSource, Manifest)} call. The caller
   * can use this to open mmap'd stores over the freshly-published files.
   *
   * @param generationNumber the generation number just written (same as the input parameter, kept
   *     for symmetry with {@link RecoveryResult})
   * @param generationDir absolute path to the finalized {@code gen-NNNNNNNNNNNNNNNN/} directory
   * @param manifest the manifest that was written to disk — identical to the input parameter
   */
  public record WriteResult(long generationNumber, Path generationDir, Manifest manifest) {}

  /**
   * Result of a {@link #recover(Path, GenerationSource, Manifest)} call. The caller opens the
   * mapped stores under the returned {@code generationDir} through an {@code Arena.ofShared()} of
   * its own.
   *
   * @param generationNumber the live generation number (points at {@code CURRENT} on return)
   * @param generationDir absolute path to the live {@code gen-NNNNNNNNNNNNNNNN/} directory
   * @param manifest the validated manifest read from disk
   * @param rewroteCurrent true iff this call had to rewrite {@code CURRENT} (either because it was
   *     missing or because its target was corrupt)
   * @param createdEmpty true iff this call bootstrapped a fresh empty gen-0 on an otherwise empty
   *     root
   */
  public record RecoveryResult(
      long generationNumber,
      Path generationDir,
      Manifest manifest,
      boolean rewroteCurrent,
      boolean createdEmpty) {}

  // ---------------------------------------------------------------------------
  // Write path.
  // ---------------------------------------------------------------------------

  /**
   * Writes a new generation directory under {@code storageRoot}, publishes it via an atomic rename,
   * and updates the collection-root {@code CURRENT} pointer to reference it. On successful return
   * the new generation is fully durable — the five fsyncs documented in the Step 4a design doc
   * §"fsync ordering summary" have all landed.
   *
   * <p>A crash during this method leaves behind at most one of: an in-flight {@code
   * .gen-NNNN.tmp/}, a fully-written {@code gen-NNNN/} that's not yet referenced by {@code
   * CURRENT}, or an in-flight {@code CURRENT.tmp}. All three are cleaned up idempotently on the
   * next call to {@link #recover(Path, GenerationSource, Manifest)}.
   *
   * @param storageRoot absolute collection root (must already exist and be a directory)
   * @param generationNumber the generation number to assign (must be {@code >= 0} and {@code <=
   *     FileFormat#MAX_GENERATION_NUMBER})
   * @param source callback that materializes the three data files into the tmp dir
   * @param manifest the manifest to write at {@code manifest.bin}; the generation number inside the
   *     manifest must match the {@code generationNumber} parameter
   * @return a {@link WriteResult} pointing at the newly-published gen directory
   * @throws IOException on any filesystem failure; partial state is cleaned up before re-throwing
   * @throws IllegalArgumentException if {@code generationNumber} is out of range or the manifest's
   *     embedded generation number disagrees with the parameter
   */
  public static WriteResult writeGeneration(
      Path storageRoot, long generationNumber, GenerationSource source, Manifest manifest)
      throws IOException {
    Objects.requireNonNull(storageRoot, "storageRoot must not be null");
    Objects.requireNonNull(source, "source must not be null");
    Objects.requireNonNull(manifest, "manifest must not be null");
    // Two-step check so the error message distinguishes "missing path" from "exists but is not a
    // directory" — both surface in user-visible stack traces during commit failures.
    if (!Files.exists(storageRoot)) {
      throw new IOException("storageRoot does not exist: " + storageRoot);
    }
    if (!Files.isDirectory(storageRoot)) {
      throw new IOException("storageRoot is not a directory: " + storageRoot);
    }
    if (manifest.generationNumber() != generationNumber) {
      throw new IllegalArgumentException(
          "manifest generation "
              + manifest.generationNumber()
              + " does not match generationNumber "
              + generationNumber);
    }

    Path tmpDir = storageRoot.resolve(FileFormat.generationTmpDirName(generationNumber));
    Path genDir = storageRoot.resolve(FileFormat.generationDirName(generationNumber));
    if (Files.exists(genDir)) {
      // Distinguish "valid prior generation" from "non-directory file squatting the slot" so the
      // error message points the operator at the right remediation. A regular file at this path
      // is almost certainly user error (or filesystem corruption); the recovery sweep ignores it.
      if (!Files.isDirectory(genDir)) {
        throw new IOException(
            "generation slot is occupied by a non-directory file: "
                + genDir
                + " (remove or rename it before re-running commit)");
      }
      throw new IOException("generation directory already exists: " + genDir);
    }
    if (Files.exists(tmpDir)) {
      // Defensive: a prior crashed commit should have been cleaned by recover(), but if the
      // caller is driving us without a recovery sweep we cannot proceed with stale state.
      throw new IOException(
          "in-flight tmp generation directory already exists (run recover() first): " + tmpDir);
    }

    Files.createDirectory(tmpDir);
    try {
      // 1. Write the data files. The GenerationSource callbacks are expected to fsync their own
      //    file contents. If any callback throws, we catch below and clean up. The graph.bin
      //    callback is only invoked when the manifest declares a graph index (currently HNSW);
      //    flat-scan generations leave no graph.bin entry on disk and the manifest's
      //    graphBinLength field carries 0 as the authoritative "no graph" signal.
      source.writeVectors(tmpDir.resolve(FileFormat.VECTORS_FILE));
      source.writeIdmap(tmpDir.resolve(FileFormat.IDMAP_FILE));
      source.writeMetadata(tmpDir.resolve(FileFormat.METADATA_FILE));
      // Skip the graph callback unless the manifest both declares a graph index type AND records
      // a non-zero graph payload length. An empty HNSW generation (bootstrap or liveCount=0 after
      // a hypothetical compaction) has nothing to write; we treat graphBinLength==0 as the
      // authoritative "no graph" signal just like FLAT, so the on-disk shape of an empty HNSW
      // generation is indistinguishable from a FLAT one.
      if (GRAPH_INDEX_TYPES.contains(manifest.indexType()) && manifest.graphBinLength() > 0L) {
        source.writeGraph(tmpDir.resolve(FileFormat.GRAPH_FILE));
      }

      // 2. Write manifest.bin (fsyncs itself via Manifest.writeTo).
      manifest.writeTo(tmpDir.resolve(FileFormat.MANIFEST_FILE));

      // 3. Fsync the tmp dir so the directory entries are durable before the rename.
      fsyncDirectory(tmpDir);
    } catch (Exception e) {
      // Best-effort cleanup of the partially-populated tmp dir. Suppress secondary failures so
      // the primary exception reaches the caller unchanged.
      try {
        deleteRecursively(tmpDir);
      } catch (IOException suppressed) {
        e.addSuppressed(suppressed);
      }
      if (e instanceof IOException io) {
        throw io;
      }
      if (e instanceof RuntimeException re) {
        throw re;
      }
      throw new IOException(e);
    }

    // 4. Atomic rename tmp → gen. After this point the generation exists on disk in its final
    //    form but is not yet referenced by CURRENT.
    Files.move(tmpDir, genDir, StandardCopyOption.ATOMIC_MOVE);
    fsyncDirectory(storageRoot);

    // 5. Atomically publish CURRENT to point at the new generation.
    writeCurrentAtomic(storageRoot, generationNumber);

    return new WriteResult(generationNumber, genDir, manifest);
  }

  // ---------------------------------------------------------------------------
  // CURRENT pointer — 8-byte little-endian int64 at <root>/CURRENT.
  // ---------------------------------------------------------------------------

  /**
   * Reads the {@code CURRENT} pointer from {@code storageRoot}. Returns {@code -1L} if the file
   * does not exist, is shorter than 8 bytes, or contains a negative generation number.
   *
   * @param storageRoot absolute collection root
   * @throws IOException on I/O failure other than {@code NoSuchFileException}
   */
  public static long readCurrent(Path storageRoot) throws IOException {
    Objects.requireNonNull(storageRoot, "storageRoot must not be null");
    Path current = storageRoot.resolve(FileFormat.CURRENT_FILE);
    try {
      byte[] bytes = Files.readAllBytes(current);
      if (bytes.length < Long.BYTES) {
        return -1L;
      }
      long n = ByteBuffer.wrap(bytes, 0, Long.BYTES).order(ByteOrder.LITTLE_ENDIAN).getLong();
      return n < 0 ? -1L : n;
    } catch (NoSuchFileException e) {
      return -1L;
    }
  }

  /**
   * Atomically updates {@code CURRENT} to reference {@code generationNumber}. Writes {@code
   * CURRENT.tmp}, fsyncs it, renames it to {@code CURRENT}, and fsyncs the parent directory. A
   * crash at any intermediate step leaves either the old {@code CURRENT} intact or a stray {@code
   * CURRENT.tmp} that {@link #recover(Path, GenerationSource, Manifest)} will clean on the next
   * open.
   *
   * <p><b>Single-process assumption.</b> The {@code deleteIfExists(CURRENT.tmp)} + {@code
   * CREATE_NEW(CURRENT.tmp)} sequence has a TOCTOU window — if a second process were driving the
   * same collection root concurrently and raced in to create {@code CURRENT.tmp} between our delete
   * and our create, the second {@code CREATE_NEW} would fail with {@link
   * java.nio.file.FileAlreadyExistsException}. {@code vectors-db} is single-process by design (the
   * facade serializes commits via its writer lock; cross-process access requires an external lock
   * that the caller installs around {@code open()}/{@code commit()}), so this window is benign in
   * supported configurations.
   *
   * @param storageRoot absolute collection root
   * @param generationNumber the generation number to publish (must be {@code >= 0} and {@code <=
   *     FileFormat#MAX_GENERATION_NUMBER})
   */
  public static void writeCurrentAtomic(Path storageRoot, long generationNumber)
      throws IOException {
    Objects.requireNonNull(storageRoot, "storageRoot must not be null");
    if (generationNumber < 0 || generationNumber > FileFormat.MAX_GENERATION_NUMBER) {
      throw new IllegalArgumentException(
          "generationNumber out of range [0, "
              + FileFormat.MAX_GENERATION_NUMBER
              + "]: "
              + generationNumber);
    }

    Path current = storageRoot.resolve(FileFormat.CURRENT_FILE);
    Path currentTmp = storageRoot.resolve(FileFormat.CURRENT_TMP_FILE);

    // A leftover CURRENT.tmp from a previous crashed attempt is harmless and safe to overwrite.
    Files.deleteIfExists(currentTmp);

    byte[] bytes = new byte[Long.BYTES];
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putLong(generationNumber);
    try (FileChannel ch =
        FileChannel.open(currentTmp, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      ByteBuffer buf = ByteBuffer.wrap(bytes);
      while (buf.hasRemaining()) {
        ch.write(buf);
      }
      ch.force(true);
    }
    Files.move(
        currentTmp, current, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    fsyncDirectory(storageRoot);
  }

  // ---------------------------------------------------------------------------
  // Recovery path.
  // ---------------------------------------------------------------------------

  /**
   * Runs the crash-recovery sweep on {@code storageRoot} and returns the resolved live generation.
   * See the class Javadoc for the full protocol. This method is idempotent — running it twice in
   * succession on a clean directory does nothing on the second call.
   *
   * @param storageRoot absolute collection root; the directory is created if missing
   * @param bootstrapSource callback used to materialize an empty {@code gen-0000000000000000/} when
   *     the root is empty (no valid generations). May be {@code null}, in which case an empty root
   *     is an error.
   * @param bootstrapManifest empty-generation manifest used alongside {@code bootstrapSource}. Must
   *     be non-null iff {@code bootstrapSource} is non-null, and its {@link
   *     Manifest#generationNumber()} must be {@code 0}.
   * @return a {@link RecoveryResult} describing the live generation
   * @throws IOException on I/O failure, or if the root is empty and no bootstrap source was given,
   *     or if every candidate {@code gen-NNNN/} has a corrupt manifest or a corrupt payload file
   */
  public static RecoveryResult recover(
      Path storageRoot, GenerationSource bootstrapSource, Manifest bootstrapManifest)
      throws IOException {
    Objects.requireNonNull(storageRoot, "storageRoot must not be null");
    if (bootstrapSource == null ^ bootstrapManifest == null) {
      throw new IllegalArgumentException(
          "bootstrapSource and bootstrapManifest must be either both null or both non-null");
    }
    if (bootstrapManifest != null && bootstrapManifest.generationNumber() != 0) {
      throw new IllegalArgumentException(
          "bootstrapManifest generationNumber must be 0, got "
              + bootstrapManifest.generationNumber());
    }

    if (!Files.exists(storageRoot)) {
      Files.createDirectories(storageRoot);
    } else if (!Files.isDirectory(storageRoot)) {
      throw new IOException("storageRoot is not a directory: " + storageRoot);
    }

    // 1. Delete every .gen-*.tmp/ directory. These are partial writes; they can never be
    //    recovered even if the data inside is intact, because the caller has no way to know
    //    whether every file was fully written before the crash.
    cleanTmpDirectories(storageRoot);

    // 2. Delete CURRENT.tmp if present — partial rename.
    Files.deleteIfExists(storageRoot.resolve(FileFormat.CURRENT_TMP_FILE));

    // 3. Try the happy path: CURRENT exists and points at a valid generation. "Valid" here means
    //    BOTH the manifest self-CRC validates AND every payload file's CRC matches the manifest's
    //    stored per-file CRCs. A manifest that parses cleanly but references a corrupt payload
    //    (bit-rot, partial truncation, user error after a restore) is treated the same way as a
    //    corrupt manifest — fall through to the descending scan and walk back to an older
    //    generation whose payload is intact.
    long currentGen = readCurrent(storageRoot);
    if (currentGen >= 0) {
      Path genDir = storageRoot.resolve(FileFormat.generationDirName(currentGen));
      Manifest manifest = tryReadManifest(genDir);
      if (manifest != null
          && manifest.generationNumber() == currentGen
          && tryVerifyPayloadCrcs(genDir, manifest)) {
        return new RecoveryResult(currentGen, genDir, manifest, false, false);
      }
    }

    // 4. CURRENT is missing or its target is corrupt (either the manifest or a payload file).
    //    Scan gen-NNNN/ directories in descending order and take the newest one whose manifest
    //    AND payload files both pass CRC validation. The walk-back here is the whole point of
    //    generation directories — it lets an operator "repair" corruption by rolling back to an
    //    intact prior generation without any manual intervention.
    List<Long> candidates = listGenerationDirectories(storageRoot);
    candidates.sort(Comparator.reverseOrder());
    for (long candidate : candidates) {
      Path genDir = storageRoot.resolve(FileFormat.generationDirName(candidate));
      Manifest manifest = tryReadManifest(genDir);
      if (manifest != null
          && manifest.generationNumber() == candidate
          && tryVerifyPayloadCrcs(genDir, manifest)) {
        writeCurrentAtomic(storageRoot, candidate);
        return new RecoveryResult(candidate, genDir, manifest, true, false);
      }
    }

    // 5. No valid generation found. Bootstrap an empty gen-0 if we have the helpers, otherwise
    //    bail out so the caller can decide what to do.
    if (bootstrapSource == null) {
      throw new IOException(
          "no valid generation found at " + storageRoot + " and no bootstrap source supplied");
    }
    WriteResult w = writeGeneration(storageRoot, 0L, bootstrapSource, bootstrapManifest);
    return new RecoveryResult(0L, w.generationDir(), w.manifest(), true, true);
  }

  // ---------------------------------------------------------------------------
  // Helpers.
  // ---------------------------------------------------------------------------

  /**
   * Returns the list of {@code gen-NNNNNNNNNNNNNNNN} sub-directories directly under {@code
   * storageRoot}, parsed back to their generation numbers. Directories whose names don't match the
   * expected pattern are silently ignored. The returned list is unsorted — callers that need a
   * specific order (e.g. {@link #recover(Path, GenerationSource, Manifest)} sorting descending)
   * apply their own comparator.
   */
  static List<Long> listGenerationDirectories(Path storageRoot) throws IOException {
    List<Long> out = new ArrayList<>();
    try (DirectoryStream<Path> stream = Files.newDirectoryStream(storageRoot)) {
      for (Path child : stream) {
        if (!Files.isDirectory(child)) {
          continue;
        }
        // getFileName() returns null for filesystem roots like "/"; the directory stream cannot
        // produce such entries (children of an existing root always have a name component) but
        // SpotBugs cannot prove that, so guard explicitly.
        Path nameComponent = child.getFileName();
        if (nameComponent == null) {
          continue;
        }
        long gen = FileFormat.parseGenerationDirName(nameComponent.toString());
        if (gen >= 0) {
          out.add(gen);
        }
      }
    }
    return out;
  }

  /**
   * Attempts to read and validate {@code manifest.bin} under {@code genDir}. Returns the parsed
   * {@link Manifest} on success, or {@code null} if the file is missing, truncated, has wrong
   * magic/version/CRC, or fails any schema check. Never throws — a corrupt manifest means "fall
   * back to the previous generation", not "abort".
   */
  static Manifest tryReadManifest(Path genDir) {
    Path manifestFile = genDir.resolve(FileFormat.MANIFEST_FILE);
    if (!Files.isRegularFile(manifestFile)) {
      return null;
    }
    try {
      return Manifest.readFrom(manifestFile);
    } catch (IOException e) {
      return null;
    }
  }

  /**
   * Verifies that every payload file referenced by {@code manifest} exists, has the exact length
   * the manifest records, and has a CRC32 that matches the manifest's stored per-file CRC. Returns
   * {@code true} on success, {@code false} if any file is missing, wrong length, or has a CRC
   * mismatch. Never throws — a corrupt payload is a "walk back to the previous generation" signal
   * for {@link #recover(Path, GenerationSource, Manifest)}, not an abort.
   *
   * <p>Files whose manifest-declared length is {@code 0} are treated as "no file" sentinels and
   * skipped entirely. This matches the commit path which never materializes an empty {@code
   * graph.bin} for flat-scan generations or for empty HNSW bootstrap generations.
   *
   * <p>Called during recovery for both the CURRENT-target happy path and the descending walk-back
   * scan, so a payload-corrupt generation behaves identically to a manifest-corrupt one: the sweep
   * continues to the next older candidate. The manifest's self-CRC has already been validated by
   * {@link #tryReadManifest} before this method is invoked, so we know the stored length/CRC values
   * are themselves intact.
   *
   * <p>On mismatch the specific file path, expected length/CRC and computed length/CRC are logged
   * at {@link Level#WARNING} so operators investigating bit-rot events have a forensic trail.
   * Without this log, recovery would walk back to the previous generation with no evidence of why
   * the newer one was rejected.
   */
  private static boolean tryVerifyPayloadCrcs(Path genDir, Manifest manifest) {
    try {
      verifyOneFile(
          genDir.resolve(FileFormat.VECTORS_FILE),
          manifest.vectorsBinLength(),
          manifest.vectorsBinCrc32());
      verifyOneFile(
          genDir.resolve(FileFormat.IDMAP_FILE),
          manifest.idmapBinLength(),
          manifest.idmapBinCrc32());
      verifyOneFile(
          genDir.resolve(FileFormat.METADATA_FILE),
          manifest.metadataBinLength(),
          manifest.metadataBinCrc32());
      verifyOneFile(
          genDir.resolve(FileFormat.GRAPH_FILE),
          manifest.graphBinLength(),
          manifest.graphBinCrc32());
      return true;
    } catch (IOException e) {
      LOGGER.log(
          Level.WARNING,
          e,
          () ->
              "payload verification failed in "
                  + genDir
                  + "; walking back to the previous generation");
      return false;
    }
  }

  /**
   * Validates that {@code file} has exactly {@code expectedLength} bytes and a CRC32 equal to
   * {@code expectedCrc}. Throws {@link IOException} with a descriptive message on any mismatch.
   * Skipped (no-op) when {@code expectedLength == 0} because the caller treats a zero-length
   * manifest entry as "this file was not materialized" (the empty-HNSW-bootstrap and flat-scan
   * cases). In the zero-length case {@code file} need not exist on disk — the commit path never
   * creates a zero-byte payload file, so this no-op is the contract that keeps {@link
   * #tryVerifyPayloadCrcs} free of per-call "does graph.bin apply here?" branching.
   *
   * <p>Uses {@link Checksums#ofFile(Path)} which streams through a 64 KiB direct byte buffer rather
   * than slurping the whole file into a heap byte array — a multi-gigabyte {@code vectors.bin} is
   * verified without allocating a heap byte array of its size.
   */
  private static void verifyOneFile(Path file, long expectedLength, long expectedCrc)
      throws IOException {
    if (expectedLength == 0L) {
      return;
    }
    long actualLength = Files.size(file);
    if (actualLength != expectedLength) {
      throw new IOException(
          "payload length mismatch at "
              + file
              + ": manifest="
              + expectedLength
              + ", actual="
              + actualLength);
    }
    long computedCrc = Checksums.ofFile(file);
    if (computedCrc != expectedCrc) {
      throw new IOException(
          String.format(
              "payload CRC mismatch at %s: manifest=0x%08x, computed=0x%08x",
              file, expectedCrc, computedCrc));
    }
  }

  /**
   * Deletes every {@code .gen-NNNNNNNNNNNNNNNN.tmp/} sub-directory directly under {@code
   * storageRoot}. The match is digit-count-strict via {@link FileFormat#parseGenerationTmpDirName}
   * so that a coincidentally-named user file like {@code .gen-.tmp} or {@code .gen-foo.tmp} is left
   * alone.
   */
  private static void cleanTmpDirectories(Path storageRoot) throws IOException {
    try (DirectoryStream<Path> stream = Files.newDirectoryStream(storageRoot)) {
      for (Path child : stream) {
        if (!Files.isDirectory(child)) {
          continue;
        }
        // See listGenerationDirectories — same SpotBugs nullability guard.
        Path nameComponent = child.getFileName();
        if (nameComponent == null) {
          continue;
        }
        if (FileFormat.parseGenerationTmpDirName(nameComponent.toString()) >= 0) {
          deleteRecursively(child);
        }
      }
    }
  }

  /**
   * Fsyncs a directory so its entries (new files, renames) become durable. On POSIX this is a
   * directory {@code fsync()}; on Windows the underlying {@code FileChannel.open(dir, READ)} raises
   * {@link AccessDeniedException} because NTFS does not expose directories as files — we swallow
   * that one specific exception class because NTFS journals directory entries synchronously anyway,
   * so the rename is durable without an explicit fsync.
   *
   * <p>We deliberately do <b>not</b> match on the exception message string. An earlier version
   * tried to swallow generic {@code "Permission denied"} text, which would have hidden a real
   * EACCES (e.g., a misconfigured collection root) on POSIX. Catching only {@link
   * AccessDeniedException} ensures we suppress only the documented Windows quirk and propagate
   * every other failure to the caller — including a true permission problem on POSIX, which would
   * cause this collection to be undurably committed and must surface immediately.
   */
  private static void fsyncDirectory(Path dir) throws IOException {
    try (FileChannel ch = FileChannel.open(dir, StandardOpenOption.READ)) {
      ch.force(true);
    } catch (AccessDeniedException e) {
      // Windows / NTFS: opening a directory read-only is not permitted. The atomic rename
      // already journaled the entry, so we skip the explicit fsync. Every other filesystem
      // error — including a real EACCES on POSIX — propagates unchanged.
    }
  }

  /**
   * Recursively deletes a directory tree. No-op if {@code root} does not exist. Safe to call on a
   * tmp dir that was partially populated — every file is deleted before the enclosing directory.
   */
  static void deleteRecursively(Path root) throws IOException {
    if (!Files.exists(root)) {
      return;
    }
    Files.walkFileTree(
        root,
        new SimpleFileVisitor<>() {
          @Override
          public FileVisitResult visitFile(Path file, BasicFileAttributes attrs)
              throws IOException {
            Files.deleteIfExists(file);
            return FileVisitResult.CONTINUE;
          }

          @Override
          public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
            if (exc != null) {
              throw exc;
            }
            Files.deleteIfExists(dir);
            return FileVisitResult.CONTINUE;
          }
        });
  }
}
