package com.integrallis.vectors.server;

import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.VectorCollectionBuilder;
import com.integrallis.vectors.db.storage.FileFormat;
import com.integrallis.vectors.db.storage.GenerationDirectory;
import com.integrallis.vectors.db.storage.Manifest;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.time.Instant;
import java.util.Objects;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Scans a {@code dataDir} on startup and reopens every persisted collection it finds into the
 * supplied {@link CollectionRegistry}.
 *
 * <p>A directory is treated as a valid collection iff its name matches the URL-safe collection-name
 * pattern and it contains a {@code CURRENT} pointer file plus the generation directory it
 * references. Everything else — stray files, unrecognised subdirectories, in-flight {@code *.tmp}
 * artifacts — is logged and skipped so one damaged collection can't take the server down.
 */
public final class CollectionDiscovery {

  private static final Logger LOG = LoggerFactory.getLogger(CollectionDiscovery.class);

  /** Must match the same charset / length as the {@code CreateCollectionRequest} validator. */
  private static final Pattern NAME_PATTERN = Pattern.compile("^[A-Za-z0-9_-]{1,128}$");

  private CollectionDiscovery() {}

  /**
   * Scans {@code dataDir} and reopens all persisted collections into {@code registry}.
   *
   * <p>No-op if {@code dataDir} is {@code null} or does not yet exist — a fresh server started with
   * {@code --data-dir} pointed at an empty directory is supported.
   *
   * @return the number of collections reopened
   */
  public static int discoverAndOpen(CollectionRegistry registry, Path dataDir) {
    Objects.requireNonNull(registry, "registry");
    if (dataDir == null || !Files.isDirectory(dataDir)) {
      return 0;
    }
    int reopened = 0;
    try (Stream<Path> subdirs = Files.list(dataDir)) {
      for (Path candidate : (Iterable<Path>) subdirs::iterator) {
        if (!Files.isDirectory(candidate)) {
          continue;
        }
        Path fileName = candidate.getFileName();
        if (fileName == null) {
          continue;
        }
        String name = fileName.toString();
        if (!NAME_PATTERN.matcher(name).matches()) {
          LOG.debug("skipping non-collection directory: {}", candidate);
          continue;
        }
        if (reopenOne(registry, name, candidate)) {
          reopened++;
        }
      }
    } catch (IOException e) {
      LOG.warn("failed to scan data dir {}: {}", dataDir, e.getMessage());
    }
    return reopened;
  }

  private static boolean reopenOne(CollectionRegistry registry, String name, Path storageRoot) {
    try {
      long gen = GenerationDirectory.readCurrent(storageRoot);
      if (gen < 0) {
        LOG.debug("skipping {}: no CURRENT pointer", storageRoot);
        return false;
      }
      Path genDir = storageRoot.resolve(FileFormat.generationDirName(gen));
      Path manifestFile = genDir.resolve(FileFormat.MANIFEST_FILE);
      if (!Files.exists(manifestFile)) {
        LOG.warn("skipping {}: manifest missing at {}", name, manifestFile);
        return false;
      }
      Manifest manifest = Manifest.readFrom(manifestFile);
      Instant createdAt = readCreationInstant(storageRoot, manifest);
      registry.reopen(
          name,
          n -> buildFromManifest(manifest, storageRoot),
          createdAt,
          manifest.generationNumber());
      LOG.info(
          "reopened collection '{}' dim={} metric={} index={} size={}",
          name,
          manifest.dimension(),
          manifest.metric(),
          manifest.indexType(),
          manifest.liveCount());
      return true;
    } catch (IOException | RuntimeException e) {
      LOG.warn("failed to reopen {}: {}", name, e.getMessage());
      return false;
    }
  }

  private static VectorCollection buildFromManifest(Manifest manifest, Path storageRoot) {
    VectorCollectionBuilder builder =
        VectorCollection.builder()
            .dimension(manifest.dimension())
            .metric(manifest.metric())
            .indexType(manifest.indexType())
            .quantizer(manifest.quantizerKind())
            .storagePath(storageRoot);
    return builder.build();
  }

  private static Instant readCreationInstant(Path storageRoot, Manifest manifest) {
    try {
      BasicFileAttributes attrs = Files.readAttributes(storageRoot, BasicFileAttributes.class);
      long millis = attrs.creationTime().toMillis();
      if (millis > 0) {
        return Instant.ofEpochMilli(millis);
      }
    } catch (IOException ignored) {
      // fall through
    }
    return Instant.ofEpochMilli(manifest.createdEpochMillis());
  }
}
