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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.VectorCollection;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.CRC32;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Migration from the version 4 manifest to the current one.
 *
 * <p>The fixtures are built by writing a real collection with this build and then
 * <b>downgrading</b> its manifests to version 4, rather than by hand-assembling a header. A
 * hand-made fixture would only prove the migrator agrees with the test author's reading of the
 * format; downgrading a manifest this build actually wrote, and then requiring the round trip to
 * return the same documents, exercises the layout the writer really produces.
 */
@Tag("unit")
class ManifestMigrationTest {

  private static final int V4_HEADER_SIZE = 164;
  private static final int V4_SELF_CRC_OFFSET = 160;

  private static final List<Document> CORPUS =
      List.of(
          Document.of("a", new float[] {1.0f, 0.0f, 0.0f}, "first"),
          Document.of("b", new float[] {0.0f, 1.0f, 0.0f}, "second"),
          Document.of("c", new float[] {0.0f, 0.0f, 1.0f}, "third"));

  /** Writes a persistent collection with this build, then rewrites its manifests as version 4. */
  private static Path oldFormatCollection(Path root) throws IOException {
    try (VectorCollection collection = open(root, false)) {
      collection.addAll(CORPUS);
      collection.commit();
    }
    int downgraded = 0;
    try (var stream = Files.list(root)) {
      for (Path generation : stream.filter(Files::isDirectory).toList()) {
        Path manifest = generation.resolve(FileFormat.MANIFEST_FILE);
        if (!Files.isRegularFile(manifest)) {
          continue;
        }
        byte[] current = Files.readAllBytes(manifest);
        byte[] old = new byte[V4_HEADER_SIZE];
        System.arraycopy(current, 0, old, 0, V4_SELF_CRC_OFFSET);
        ByteBuffer buffer = ByteBuffer.wrap(old).order(ByteOrder.LITTLE_ENDIAN);
        buffer.putInt(4, 4);
        buffer.putInt(8, V4_HEADER_SIZE);
        CRC32 self = new CRC32();
        self.update(old, 0, V4_SELF_CRC_OFFSET);
        buffer.putInt(V4_SELF_CRC_OFFSET, (int) self.getValue());
        Files.write(manifest, old);
        downgraded++;
      }
    }
    assertThat(downgraded).describedAs("fixture must contain a manifest to downgrade").isPositive();
    return root;
  }

  private static VectorCollection open(Path root, boolean migrate) {
    return VectorCollection.builder()
        .dimension(3)
        .metric(SimilarityFunction.COSINE)
        .indexType(IndexType.FLAT)
        .storagePath(root.toAbsolutePath())
        .migrateOlderFormats(migrate)
        .build();
  }

  @Test
  void refusesAnOlderCollectionByDefaultAndSaysHowToMigrateIt(@TempDir Path root) throws Exception {
    oldFormatCollection(root);

    assertThatThrownBy(() -> open(root, false).close())
        .rootCause()
        .isInstanceOf(IOException.class)
        // The point of the message: name both versions and the way forward, not the filesystem.
        .hasMessageContaining("manifest format version 4")
        .hasMessageContaining("reads version " + FileFormat.VERSION_MANIFEST)
        .hasMessageContaining("ManifestMigration");
  }

  @Test
  void inspectReportsWhatItWouldDoWithoutTouchingAnything(@TempDir Path root) throws Exception {
    oldFormatCollection(root);
    byte[] before =
        Files.readAllBytes(root.resolve(latestGeneration(root)).resolve(FileFormat.MANIFEST_FILE));

    List<ManifestMigration.Result> results = ManifestMigration.inspect(root);

    assertThat(results).isNotEmpty();
    // inspect() writes nothing, so it must not claim MIGRATED. Reporting the same value for "would
    // migrate" and "did migrate" is how a caller ends up believing a dry run upgraded something.
    assertThat(results)
        .anySatisfy(
            r -> assertThat(r.outcome()).isEqualTo(ManifestMigration.Outcome.NEEDS_MIGRATION));
    assertThat(results)
        .noneSatisfy(r -> assertThat(r.outcome()).isEqualTo(ManifestMigration.Outcome.MIGRATED));
    assertThat(ManifestMigration.migrationAvailable(root)).isTrue();
    assertThat(
            Files.readAllBytes(
                root.resolve(latestGeneration(root)).resolve(FileFormat.MANIFEST_FILE)))
        .describedAs("inspect must not write")
        .isEqualTo(before);
  }

  @Test
  void migratesInPlaceAndReturnsEveryDocument(@TempDir Path root) throws Exception {
    oldFormatCollection(root);

    List<ManifestMigration.Result> results = ManifestMigration.migrate(root);
    assertThat(results)
        .filteredOn(r -> r.outcome() == ManifestMigration.Outcome.MIGRATED)
        .isNotEmpty()
        .allSatisfy(r -> assertThat(r.fromVersion()).isEqualTo(4));
    // And migrate() must not report NEEDS_MIGRATION for something it just rewrote.
    assertThat(results)
        .noneSatisfy(
            r -> assertThat(r.outcome()).isEqualTo(ManifestMigration.Outcome.NEEDS_MIGRATION));

    try (VectorCollection collection = open(root, false)) {
      assertThat(collection.size()).isEqualTo(CORPUS.size());
      assertThat(collection.documents())
          .extracting(Document::id)
          .containsExactlyInAnyOrder("a", "b", "c");
      // Searchable, not merely openable.
      assertThat(
              collection
                  .search(SearchRequest.builder(new float[] {1.0f, 0.0f, 0.0f}, 1).build())
                  .hits()
                  .get(0)
                  .id())
          .isEqualTo("a");
    }
  }

  @Test
  void keepsThePreviousManifestSoTheCollectionCanGoBack(@TempDir Path root) throws Exception {
    oldFormatCollection(root);
    Path generation = root.resolve(latestGeneration(root));
    byte[] original = Files.readAllBytes(generation.resolve(FileFormat.MANIFEST_FILE));

    ManifestMigration.migrate(root);

    Path backup = generation.resolve(FileFormat.MANIFEST_FILE + ".v4.bak");
    assertThat(backup).exists();
    assertThat(Files.readAllBytes(backup)).isEqualTo(original);
    assertThat(Files.size(generation.resolve(FileFormat.MANIFEST_FILE)))
        .isEqualTo(Manifest.HEADER_SIZE);
  }

  @Test
  void openingWithMigrationEnabledUpgradesAndOpens(@TempDir Path root) throws Exception {
    oldFormatCollection(root);

    try (VectorCollection collection = open(root, true)) {
      assertThat(collection.size()).isEqualTo(CORPUS.size());
    }
    assertThat(ManifestMigration.migrationAvailable(root)).isFalse();
  }

  @Test
  void migratingTwiceIsANoOp(@TempDir Path root) throws Exception {
    oldFormatCollection(root);
    ManifestMigration.migrate(root);
    Path manifest = root.resolve(latestGeneration(root)).resolve(FileFormat.MANIFEST_FILE);
    byte[] once = Files.readAllBytes(manifest);

    List<ManifestMigration.Result> second = ManifestMigration.migrate(root);

    assertThat(second)
        .allSatisfy(
            r -> assertThat(r.outcome()).isEqualTo(ManifestMigration.Outcome.ALREADY_CURRENT));
    assertThat(Files.readAllBytes(manifest)).isEqualTo(once);
  }

  @Test
  void refusesToMigrateACorruptHeaderRatherThanLaunderIt(@TempDir Path root) throws Exception {
    oldFormatCollection(root);
    Path manifest = root.resolve(latestGeneration(root)).resolve(FileFormat.MANIFEST_FILE);
    byte[] corrupt = Files.readAllBytes(manifest);
    // Flip a payload length. The version 4 self CRC now disagrees, which is exactly the case a
    // migration must not paper over by recomputing a fresh CRC over bad bytes.
    corrupt[56] = (byte) (corrupt[56] ^ 0xFF);
    Files.write(manifest, corrupt);

    assertThatThrownBy(() -> ManifestMigration.migrate(root))
        .isInstanceOf(IOException.class)
        .hasMessageContaining("self CRC")
        .hasMessageContaining("must not be migrated");
  }

  @Test
  void leavesANewerManifestAloneInsteadOfDowngradingIt(@TempDir Path root) throws Exception {
    try (VectorCollection collection = open(root, false)) {
      collection.addAll(CORPUS);
      collection.commit();
    }
    Path manifest = root.resolve(latestGeneration(root)).resolve(FileFormat.MANIFEST_FILE);
    byte[] future = Files.readAllBytes(manifest);
    ByteBuffer.wrap(future)
        .order(ByteOrder.LITTLE_ENDIAN)
        .putInt(4, FileFormat.VERSION_MANIFEST + 1);
    Files.write(manifest, future);

    List<ManifestMigration.Result> results = ManifestMigration.migrate(root);

    assertThat(results)
        .anySatisfy(
            r ->
                assertThat(r.outcome()).isEqualTo(ManifestMigration.Outcome.NEWER_THAN_THIS_BUILD));
    assertThat(Files.readAllBytes(manifest)).describedAs("must not downgrade").isEqualTo(future);
  }

  private static String latestGeneration(Path root) throws IOException {
    try (var stream = Files.list(root)) {
      return stream
          .filter(Files::isDirectory)
          .map(p -> p.getFileName().toString())
          .filter(n -> n.startsWith(FileFormat.GENERATION_DIR_PREFIX))
          .sorted()
          .reduce((a, b) -> b)
          .orElseThrow(() -> new IOException("no generation directory under " + root));
    }
  }
}
