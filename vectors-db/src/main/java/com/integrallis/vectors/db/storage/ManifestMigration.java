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
package com.integrallis.vectors.db.storage;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.zip.CRC32;

/**
 * Upgrades a collection's manifests in place, without re-embedding anything.
 *
 * <p>Version 5 added an embedding recipe anchor: a present flag at offset 160 and a 32-byte hash
 * after it, which pushed the self CRC from 160 to 196 and the header from 164 to 200 bytes. Nothing
 * else moved. Every field a version 4 manifest carries keeps its offset, and <b>no payload file
 * changed format at all</b> — {@code VERSION_METADATA}, {@code VERSION_IDMAP}, {@code
 * VERSION_QUANTIZED}, {@code VERSION_GRAPH} and {@code VERSION_TOMBSTONES} are all still 1, and the
 * content hash that version 5 introduced is derived from stored text on read rather than persisted.
 *
 * <p>So the upgrade is a few hundred bytes per generation, not a rebuild. That distinction matters
 * because the alternative is re-running an embedding model over the whole corpus: migrating the
 * bundled 1,929-prompt router index this way produced <b>per-item search results identical to a
 * full rebuild</b> on all 481 held-out prompts, against fifteen minutes to re-embed it.
 *
 * <p><b>Not automatic.</b> Rewriting a file inside someone's collection is not something to do
 * because they happened to open it, so this never runs on its own — a caller asks for it, either
 * through {@link com.integrallis.vectors.db.VectorCollectionBuilder#migrateOlderFormats(boolean)}
 * or by calling {@link #migrate(Path)} directly. The previous manifest is kept beside the new one
 * as {@code manifest.bin.v<n>.bak} so the collection can be handed back to the version that wrote
 * it.
 *
 * <p><b>One direction only.</b> A manifest from a <i>newer</i> build cannot be downgraded: it may
 * carry fields this one does not know how to preserve, and dropping them silently is worse than
 * refusing. Those are reported and skipped.
 */
public final class ManifestMigration {

  /** Header size of a version 4 manifest, whose self CRC sits at its final four bytes. */
  private static final int V4_HEADER_SIZE = 164;

  /** Offset of the version 4 self CRC, which is also the length it covers. */
  private static final int V4_SELF_CRC_OFFSET = 160;

  /** The oldest manifest version this class can read. */
  public static final int OLDEST_SUPPORTED = 4;

  private ManifestMigration() {}

  /** What happened to one generation. */
  public enum Outcome {
    /** Rewritten from an older version to the current one. Only {@link #migrate} returns this. */
    MIGRATED,
    /**
     * Older than this build and readable, but nothing was written.
     *
     * <p>What {@link #inspect} returns where {@link #migrate} would return {@link #MIGRATED}. Kept
     * separate because one value for both would let a caller treat a report as a completed upgrade,
     * which is exactly the mistake that is easy to make and impossible to see afterwards.
     */
    NEEDS_MIGRATION,
    /** Already at the current version; untouched. */
    ALREADY_CURRENT,
    /** Written by a newer build; untouched, because downgrading could drop fields. */
    NEWER_THAN_THIS_BUILD,
    /** Too old for this build to read, or not a manifest; untouched. */
    UNSUPPORTED
  }

  /**
   * One generation's migration result.
   *
   * @param generation the generation directory
   * @param fromVersion the version found on disk, or -1 if it could not be read
   * @param outcome what was done
   */
  public record Result(Path generation, int fromVersion, Outcome outcome) {

    /** Compact constructor. */
    public Result {
      Objects.requireNonNull(generation, "generation");
      Objects.requireNonNull(outcome, "outcome");
    }
  }

  /**
   * Reports what {@link #migrate} would do, changing nothing.
   *
   * @param collectionRoot the collection's storage path
   * @return one result per generation directory, oldest first
   * @throws IOException if the collection root cannot be listed
   */
  public static List<Result> inspect(Path collectionRoot) throws IOException {
    return run(collectionRoot, false);
  }

  /**
   * Upgrades every generation that this build can read and that is older than the current version.
   *
   * <p>Safe to run twice: a generation already at the current version is left alone.
   *
   * @param collectionRoot the collection's storage path
   * @return one result per generation directory, oldest first
   * @throws IOException if a manifest cannot be read or replaced
   */
  public static List<Result> migrate(Path collectionRoot) throws IOException {
    return run(collectionRoot, true);
  }

  /**
   * Whether any generation under {@code collectionRoot} is older than this build writes.
   *
   * @param collectionRoot the collection's storage path
   * @return true if {@link #migrate} would rewrite something
   * @throws IOException if the collection root cannot be listed
   */
  public static boolean migrationAvailable(Path collectionRoot) throws IOException {
    return inspect(collectionRoot).stream().anyMatch(r -> r.outcome() == Outcome.NEEDS_MIGRATION);
  }

  private static List<Result> run(Path collectionRoot, boolean apply) throws IOException {
    Objects.requireNonNull(collectionRoot, "collectionRoot");
    List<Result> results = new ArrayList<>();
    List<Path> generations = new ArrayList<>();
    if (!Files.isDirectory(collectionRoot)) {
      return results;
    }
    try (var stream = Files.list(collectionRoot)) {
      stream
          .filter(Files::isDirectory)
          .filter(p -> p.getFileName().toString().startsWith(FileFormat.GENERATION_DIR_PREFIX))
          .sorted()
          .forEach(generations::add);
    }
    for (Path generation : generations) {
      Integer version = GenerationDirectory.peekManifestVersion(generation);
      if (version == null) {
        results.add(new Result(generation, -1, Outcome.UNSUPPORTED));
      } else if (version == FileFormat.VERSION_MANIFEST) {
        results.add(new Result(generation, version, Outcome.ALREADY_CURRENT));
      } else if (version > FileFormat.VERSION_MANIFEST) {
        results.add(new Result(generation, version, Outcome.NEWER_THAN_THIS_BUILD));
      } else if (version < OLDEST_SUPPORTED) {
        results.add(new Result(generation, version, Outcome.UNSUPPORTED));
      } else {
        if (apply) {
          upgradeV4ToV5(generation, version);
        }
        results.add(
            new Result(generation, version, apply ? Outcome.MIGRATED : Outcome.NEEDS_MIGRATION));
      }
    }
    return results;
  }

  /**
   * Rewrites one version 4 manifest as version 5.
   *
   * <p>The old header is copied forward verbatim up to the recipe fields, so every value the
   * collection already carried — offsets, lengths, per-file CRCs, the creation timestamp — survives
   * exactly. Only the version, the header length, the new recipe region and the self CRC are
   * written. The recipe flag is 0: a collection written before recipes existed has no provenance to
   * claim, and inventing one here would make {@code attested} meaningless.
   */
  private static void upgradeV4ToV5(Path generation, int fromVersion) throws IOException {
    Path manifestFile = generation.resolve(FileFormat.MANIFEST_FILE);
    byte[] old = Files.readAllBytes(manifestFile);
    if (old.length < V4_HEADER_SIZE) {
      throw new IOException(
          "manifest at " + manifestFile + " is " + old.length + " bytes, too short for version 4");
    }

    ByteBuffer oldBuf = ByteBuffer.wrap(old).order(ByteOrder.LITTLE_ENDIAN);
    int storedSelfCrc = oldBuf.getInt(V4_SELF_CRC_OFFSET);
    CRC32 check = new CRC32();
    check.update(old, 0, V4_SELF_CRC_OFFSET);
    if ((int) check.getValue() != storedSelfCrc) {
      // Refuse rather than propagate: migrating a corrupt header would launder the corruption into
      // a manifest that passes its own checksum.
      throw new IOException(
          "manifest at "
              + manifestFile
              + " fails its own version "
              + fromVersion
              + " self CRC; it is corrupt and must not be migrated");
    }

    byte[] upgraded = new byte[Manifest.HEADER_SIZE];
    System.arraycopy(old, 0, upgraded, 0, Manifest.RECIPE_FLAG_OFFSET);
    ByteBuffer out = ByteBuffer.wrap(upgraded).order(ByteOrder.LITTLE_ENDIAN);
    out.putInt(4, FileFormat.VERSION_MANIFEST);
    out.putInt(8, Manifest.HEADER_SIZE);
    out.putInt(Manifest.RECIPE_FLAG_OFFSET, 0);
    // Bytes RECIPE_HASH_OFFSET .. SELF_CRC_OFFSET stay zero: no recipe, so no hash to anchor.
    CRC32 self = new CRC32();
    self.update(upgraded, 0, Manifest.SELF_CRC_OFFSET);
    out.putInt(Manifest.SELF_CRC_OFFSET, (int) self.getValue());

    // Keep the original so the collection can go back to the build that wrote it.
    Path backup = generation.resolve(FileFormat.MANIFEST_FILE + ".v" + fromVersion + ".bak");
    if (!Files.exists(backup)) {
      Files.write(backup, old, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
      fsyncFile(backup);
    }

    Path tmp = generation.resolve(FileFormat.MANIFEST_FILE + ".migrating");
    Files.write(
        tmp,
        upgraded,
        StandardOpenOption.CREATE,
        StandardOpenOption.TRUNCATE_EXISTING,
        StandardOpenOption.WRITE);
    fsyncFile(tmp);
    Files.move(
        tmp, manifestFile, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    fsyncDirectory(generation);
  }

  /**
   * Batch entry point: reports or upgrades the collections under one or more paths.
   *
   * <pre>
   *   java -cp vectors-db.jar com.integrallis.vectors.db.storage.ManifestMigration [--apply] DIR...
   * </pre>
   *
   * <p>Each {@code DIR} is either a collection root or a directory holding several of them, which
   * is the shape Studio's data directory already has. Detection is by whether the directory itself
   * contains generation directories.
   *
   * <p><b>Reports by default.</b> {@code --apply} is required to write anything, so the first run a
   * user makes cannot be the destructive one. Exit status is 0 when nothing is left needing
   * migration, 1 when a report found work to do, and 2 on usage error or failure — so a dry run can
   * gate a deployment.
   *
   * @param args optional {@code --apply} followed by one or more directories
   * @throws IOException if a collection cannot be read or rewritten
   */
  public static void main(String[] args) throws IOException {
    List<String> paths = new ArrayList<>();
    boolean apply = false;
    for (String argument : args) {
      if ("--apply".equals(argument)) {
        apply = true;
      } else if ("--dry-run".equals(argument)) {
        apply = false;
      } else if (argument.startsWith("-")) {
        System.err.println("unknown option: " + argument);
        System.exit(2);
      } else {
        paths.add(argument);
      }
    }
    if (paths.isEmpty()) {
      System.err.println(
          "usage: ManifestMigration [--apply] DIR...\n"
              + "  Reports what would change unless --apply is given.\n"
              + "  DIR may be a collection root or a directory of collection roots.");
      System.exit(2);
    }

    int pending = 0;
    int migrated = 0;
    for (String path : paths) {
      for (Path root : collectionRoots(Path.of(path))) {
        List<Result> results = apply ? migrate(root) : inspect(root);
        for (Result result : results) {
          if (result.outcome() == Outcome.ALREADY_CURRENT) {
            continue;
          }
          System.out.printf(
              "%-16s v%-3s %s%n",
              result.outcome(),
              result.fromVersion() < 0 ? "?" : Integer.toString(result.fromVersion()),
              result.generation());
          if (result.outcome() == Outcome.MIGRATED) {
            migrated++;
          } else if (result.outcome() == Outcome.NEEDS_MIGRATION) {
            pending++;
          }
        }
      }
    }
    if (apply) {
      System.out.printf(
          "migrated %d generation(s) to manifest v%d%n", migrated, FileFormat.VERSION_MANIFEST);
      System.exit(0);
    }
    System.out.printf(
        "%d generation(s) would be migrated to manifest v%d; re-run with --apply%n",
        pending, FileFormat.VERSION_MANIFEST);
    System.exit(pending == 0 ? 0 : 1);
  }

  /**
   * Finds the collections under a path.
   *
   * <p>A directory is a collection root when it holds generation directories; otherwise its
   * subdirectories are searched for them. That covers both "migrate this collection" and "migrate
   * everything in this data directory" without the caller having to say which it meant.
   *
   * <p>Public because every migration entry point needs it — the CLI here, the Spring Batch reader,
   * the Jakarta Batch batchlet — and three copies of a directory-scanning rule is three chances for
   * them to disagree about what counts as a collection.
   *
   * @param path a collection root, or a directory containing collection roots
   * @return the collection roots found, in name order; empty if {@code path} is not a directory
   * @throws IOException if the directory cannot be listed
   */
  public static List<Path> collectionRoots(Path path) throws IOException {
    List<Path> roots = new ArrayList<>();
    if (!Files.isDirectory(path)) {
      return roots;
    }
    if (hasGenerations(path)) {
      roots.add(path);
      return roots;
    }
    try (var stream = Files.list(path)) {
      for (Path child : stream.filter(Files::isDirectory).sorted().toList()) {
        if (hasGenerations(child)) {
          roots.add(child);
        }
      }
    }
    return roots;
  }

  private static boolean hasGenerations(Path path) throws IOException {
    try (var stream = Files.list(path)) {
      return stream
          .filter(Files::isDirectory)
          .anyMatch(p -> p.getFileName().toString().startsWith(FileFormat.GENERATION_DIR_PREFIX));
    }
  }

  private static void fsyncFile(Path file) throws IOException {
    try (FileChannel channel = FileChannel.open(file, StandardOpenOption.WRITE)) {
      channel.force(true);
    }
  }

  private static void fsyncDirectory(Path directory) {
    try (FileChannel dir = FileChannel.open(directory, StandardOpenOption.READ)) {
      dir.force(true);
    } catch (IOException ignored) {
      // Some filesystems refuse to open a directory for fsync. The atomic rename still holds; this
      // only weakens durability across a power loss, so it is not worth failing a migration over.
    }
  }
}
