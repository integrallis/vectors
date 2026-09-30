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
package com.integrallis.vectors.jakarta.batch;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.storage.FileFormat;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.CRC32;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The JSR-352 batchlet.
 *
 * <p>Exercised through the string-valued batch properties a container would inject, not only
 * through the typed constructor, because the string path is where "apply" can be misread and the
 * one a job's JSL actually uses.
 */
@Tag("unit")
class ManifestMigrationBatchletTest {

  private static final int V4_HEADER_SIZE = 164;
  private static final int V4_SELF_CRC_OFFSET = 160;

  private static void writeOldCollection(Path root, String id) throws IOException {
    try (VectorCollection collection =
        VectorCollection.builder()
            .dimension(3)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .storagePath(root.toAbsolutePath())
            .build()) {
      collection.add(Document.of(id, new float[] {1.0f, 0.0f, 0.0f}, id));
      collection.commit();
    }
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
      }
    }
  }

  /** As the container builds it: no-arg constructor, then injected string properties. */
  private static ManifestMigrationBatchlet injected(String basePath, String apply) {
    ManifestMigrationBatchlet batchlet = new ManifestMigrationBatchlet();
    batchlet.basePath = basePath;
    batchlet.apply = apply;
    return batchlet;
  }

  @Test
  void migratesEveryCollectionAndReportsMigrated(@TempDir Path base) throws Exception {
    writeOldCollection(base.resolve("alpha"), "a");
    writeOldCollection(base.resolve("beta"), "b");

    String status = injected(base.toString(), "true").process();

    assertThat(status).isEqualTo(ManifestMigrationBatchlet.STATUS_MIGRATED);
    assertThat(injected(base.toString(), "false").process())
        .describedAs("a second pass has nothing left to do")
        .isEqualTo(ManifestMigrationBatchlet.STATUS_CLEAN);
  }

  @Test
  void aDryRunReportsPendingAndNeverMigrated(@TempDir Path base) throws Exception {
    writeOldCollection(base, "a");
    byte[] before = Files.readAllBytes(manifestOf(base));

    // The defect this pins: a dry run used to return MIGRATED. A JSL branching on that would
    // conclude the migration had happened while nothing was written.
    String status = injected(base.toString(), null).process();

    assertThat(status).isEqualTo(ManifestMigrationBatchlet.STATUS_PENDING);
    assertThat(status).isNotEqualTo(ManifestMigrationBatchlet.STATUS_MIGRATED);
    assertThat(Files.readAllBytes(manifestOf(base)))
        .describedAs("an absent apply property must not rewrite anything")
        .isEqualTo(before);
  }

  @Test
  void migratedIsReturnedOnlyWhenSomethingWasActuallyWritten(@TempDir Path base) throws Exception {
    writeOldCollection(base, "a");

    assertThat(injected(base.toString(), "false").process())
        .isEqualTo(ManifestMigrationBatchlet.STATUS_PENDING);
    assertThat(injected(base.toString(), "true").process())
        .isEqualTo(ManifestMigrationBatchlet.STATUS_MIGRATED);
    // And once there is nothing left, a write run is CLEAN rather than MIGRATED.
    assertThat(injected(base.toString(), "true").process())
        .isEqualTo(ManifestMigrationBatchlet.STATUS_CLEAN);
  }

  @Test
  void anUnrecognisedApplyValueFailsInsteadOfSilentlyDryRunning(@TempDir Path base)
      throws Exception {
    writeOldCollection(base, "a");

    // "yes" used to mean false, so an operator who typed it got a dry run that reported MIGRATED
    // and believed the collection had been upgraded. Refusing is the only safe reading.
    for (String bad : new String[] {"yes", "1", "y", "on", "ture", "truthy"}) {
      assertThatThrownBy(() -> injected(base.toString(), bad).process())
          .describedAs("apply=%s must be refused, not read as false", bad)
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("apply")
          .hasMessageContaining(bad);
    }
  }

  @Test
  void anAbsentOrBlankApplyIsFalseRatherThanAnError(@TempDir Path base) throws Exception {
    writeOldCollection(base, "a");
    byte[] before = Files.readAllBytes(manifestOf(base));

    // Absent is a job that did not ask to write, which is legitimate. Blank is the same thing in
    // XML.
    assertThat(injected(base.toString(), null).process())
        .isEqualTo(ManifestMigrationBatchlet.STATUS_PENDING);
    assertThat(injected(base.toString(), "").process())
        .isEqualTo(ManifestMigrationBatchlet.STATUS_PENDING);
    assertThat(injected(base.toString(), "   ").process())
        .isEqualTo(ManifestMigrationBatchlet.STATUS_PENDING);
    assertThat(Files.readAllBytes(manifestOf(base))).isEqualTo(before);
  }

  @Test
  void acceptsTrueWithSurroundingWhitespaceAndAnyCase(@TempDir Path base) throws Exception {
    writeOldCollection(base, "a");

    assertThat(injected(base.toString(), " True ").process())
        .isEqualTo(ManifestMigrationBatchlet.STATUS_MIGRATED);
    assertThat(injected(base.toString(), "false").process())
        .isEqualTo(ManifestMigrationBatchlet.STATUS_CLEAN);
  }

  @Test
  void reportsAttentionForSomethingAMigrationCannotFix(@TempDir Path base) throws Exception {
    writeOldCollection(base, "a");
    injected(base.toString(), "true").process();
    Path manifest = manifestOf(base);
    byte[] future = Files.readAllBytes(manifest);
    ByteBuffer.wrap(future)
        .order(ByteOrder.LITTLE_ENDIAN)
        .putInt(4, FileFormat.VERSION_MANIFEST + 1);
    Files.write(manifest, future);

    // Not CLEAN and not MIGRATED: a JSL needs to be able to branch on this one.
    assertThat(injected(base.toString(), "true").process())
        .isEqualTo(ManifestMigrationBatchlet.STATUS_ATTENTION);
  }

  @Test
  void refusesToRunWithoutABasePath() {
    assertThatThrownBy(() -> injected(null, "true").process())
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("basePath");
    assertThatThrownBy(() -> injected("   ", "true").process())
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("basePath");
  }

  @Test
  void theTypedConstructorAgreesWithTheInjectedForm(@TempDir Path base) throws Exception {
    writeOldCollection(base, "a");

    assertThat(new ManifestMigrationBatchlet(base, true).process())
        .isEqualTo(ManifestMigrationBatchlet.STATUS_MIGRATED);
    assertThat(new ManifestMigrationBatchlet(base, false).process())
        .isEqualTo(ManifestMigrationBatchlet.STATUS_CLEAN);
    assertThat(new ManifestMigrationBatchlet(base, false).process())
        .describedAs("already migrated above, so a dry run now finds nothing pending")
        .isEqualTo(ManifestMigrationBatchlet.STATUS_CLEAN);
  }

  @Test
  void reportsCleanForADirectoryHoldingNoCollections(@TempDir Path base) throws Exception {
    Files.createDirectories(base.resolve("not-a-collection"));

    assertThat(injected(base.toString(), "true").process())
        .isEqualTo(ManifestMigrationBatchlet.STATUS_CLEAN);
  }

  private static Path manifestOf(Path root) throws IOException {
    try (var stream = Files.list(root)) {
      return stream
          .filter(Files::isDirectory)
          .filter(p -> p.getFileName().toString().startsWith(FileFormat.GENERATION_DIR_PREFIX))
          .sorted()
          .reduce((a, b) -> b)
          .orElseThrow(() -> new IOException("no generation under " + root))
          .resolve(FileFormat.MANIFEST_FILE);
    }
  }
}
