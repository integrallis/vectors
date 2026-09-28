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
package com.integrallis.vectors.spring.batch;

import static org.assertj.core.api.Assertions.assertThat;

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
import java.util.List;
import java.util.zip.CRC32;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The Spring Batch adapter over several collections at once.
 *
 * <p>Fixtures are made by writing real collections with this build and downgrading their manifests,
 * so the adapter is exercised against the layout the writer actually produces rather than a
 * hand-assembled header.
 */
@Tag("unit")
class ManifestMigrationBatchTest {

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

  @Test
  void readerFindsEveryCollectionUnderAParentDirectory(@TempDir Path base) throws Exception {
    writeOldCollection(base.resolve("alpha"), "a");
    writeOldCollection(base.resolve("beta"), "b");
    Files.createDirectories(base.resolve("not-a-collection"));

    CollectionRootItemReader reader = new CollectionRootItemReader(base);
    List<Path> read = new java.util.ArrayList<>();
    for (Path p = reader.read(); p != null; p = reader.read()) {
      read.add(p);
    }

    assertThat(read).extracting(p -> p.getFileName().toString()).containsExactly("alpha", "beta");
    // The directory with no generations is not a collection and must not be offered as one.
    assertThat(read).noneMatch(p -> p.getFileName().toString().equals("not-a-collection"));
  }

  @Test
  void readerTreatsASingleCollectionRootAsOneItem(@TempDir Path base) throws Exception {
    writeOldCollection(base, "solo");

    CollectionRootItemReader reader = new CollectionRootItemReader(base);

    assertThat(reader.read()).isEqualTo(base);
    assertThat(reader.read()).isNull();
  }

  @Test
  void processorReportsWithoutWritingByDefault(@TempDir Path base) throws Exception {
    writeOldCollection(base, "a");
    byte[] before = Files.readAllBytes(manifestOf(base));

    MigrationReport report = new ManifestMigrationItemProcessor().process(base);

    assertThat(report.applied()).isFalse();
    assertThat(report.migrated()).isPositive();
    assertThat(report.clean()).isFalse();
    assertThat(Files.readAllBytes(manifestOf(base)))
        .describedAs("the default processor must not write")
        .isEqualTo(before);
  }

  @Test
  void processorMigratesWhenAsked(@TempDir Path base) throws Exception {
    writeOldCollection(base, "a");

    MigrationReport report = new ManifestMigrationItemProcessor(true).process(base);

    assertThat(report.applied()).isTrue();
    assertThat(report.migrated()).isPositive();
    assertThat(new ManifestMigrationItemProcessor().process(base).clean()).isTrue();
  }

  @Test
  void taskletMigratesEveryCollectionAndCountsTheGenerations(@TempDir Path base) throws Exception {
    writeOldCollection(base.resolve("alpha"), "a");
    writeOldCollection(base.resolve("beta"), "b");

    var contribution =
        new org.springframework.batch.core.StepContribution(
            org.springframework.batch.core.StepExecution.class
                .getDeclaredConstructor(
                    String.class, org.springframework.batch.core.JobExecution.class)
                .newInstance("migrate", new org.springframework.batch.core.JobExecution(1L)));

    new ManifestMigrationTasklet(base, true).execute(contribution, chunkContext());

    assertThat(contribution.getWriteCount()).isPositive();
    assertThat(new ManifestMigrationItemProcessor().process(base.resolve("alpha")).clean())
        .isTrue();
    assertThat(new ManifestMigrationItemProcessor().process(base.resolve("beta")).clean()).isTrue();
  }

  @Test
  void reportSeparatesWhatAMigrationCannotFix(@TempDir Path base) throws Exception {
    writeOldCollection(base, "a");
    // A generation from a newer build: another run will never help, so it is not merely "not
    // clean".
    Path manifest = manifestOf(base);
    byte[] restored = Files.readAllBytes(manifest);
    new ManifestMigrationItemProcessor(true).process(base);
    byte[] future = Files.readAllBytes(manifest);
    ByteBuffer.wrap(future)
        .order(ByteOrder.LITTLE_ENDIAN)
        .putInt(4, FileFormat.VERSION_MANIFEST + 1);
    Files.write(manifest, future);

    MigrationReport report = new ManifestMigrationItemProcessor().process(base);

    assertThat(report.needsAttention()).isTrue();
    assertThat(report.newerThanThisBuild()).isPositive();
    assertThat(report.migrated()).isZero();
    assertThat(restored).isNotEmpty();
  }

  @Test
  void theReportReadsAsSomethingAnOperatorCanAct(@TempDir Path base) throws Exception {
    writeOldCollection(base, "a");

    // toString is what lands in a job log, so it has to distinguish a report from a completed run.
    MigrationReport dryRun = new ManifestMigrationItemProcessor().process(base);
    assertThat(dryRun).asString().contains("would migrate").doesNotContain("migrated 0");
    assertThat(dryRun).asString().contains(base.toString());

    MigrationReport applied = new ManifestMigrationItemProcessor(true).process(base);
    assertThat(applied).asString().startsWith("migrated ").doesNotContain("would migrate");
  }

  @Test
  void theReportNamesWhatAFurtherRunCannotFix(@TempDir Path base) throws Exception {
    writeOldCollection(base, "a");
    new ManifestMigrationItemProcessor(true).process(base);
    Path manifest = manifestOf(base);
    byte[] future = Files.readAllBytes(manifest);
    ByteBuffer.wrap(future)
        .order(ByteOrder.LITTLE_ENDIAN)
        .putInt(4, FileFormat.VERSION_MANIFEST + 1);
    Files.write(manifest, future);

    MigrationReport report = new ManifestMigrationItemProcessor().process(base);

    assertThat(report.needsAttention()).isTrue();
    assertThat(report.clean()).isFalse();
    assertThat(report).asString().contains("NEWER than this build");
  }

  @Test
  void aCleanCollectionSaysSoAndNeedsNoAttention(@TempDir Path base) throws Exception {
    writeOldCollection(base, "a");
    new ManifestMigrationItemProcessor(true).process(base);

    MigrationReport report = new ManifestMigrationItemProcessor().process(base);

    assertThat(report.clean()).isTrue();
    assertThat(report.needsAttention()).isFalse();
    assertThat(report.migrated()).isZero();
    assertThat(report.alreadyCurrent()).isPositive();
    assertThat(report).asString().contains("already current");
  }

  @Test
  void theReaderReportsWhatItFound(@TempDir Path base) throws Exception {
    writeOldCollection(base.resolve("alpha"), "a");
    writeOldCollection(base.resolve("beta"), "b");
    CollectionRootItemReader reader = new CollectionRootItemReader(base);

    // Empty before the first read: the scan is deferred, and saying otherwise would be a lie a
    // listener could act on.
    assertThat(reader.discoveredRoots()).isEmpty();
    reader.read();

    assertThat(reader.discoveredRoots())
        .extracting(p -> p.getFileName().toString())
        .containsExactly("alpha", "beta");
  }

  @Test
  void theReaderFailsLoudlyOnAnUnreadableBase(@TempDir Path tmp) throws Exception {
    Path missing = tmp.resolve("no-such-directory");
    CollectionRootItemReader reader = new CollectionRootItemReader(missing);

    // Not a directory is not an error: it holds no collections, so the step has nothing to do.
    assertThat(reader.read()).isNull();
    assertThat(reader.discoveredRoots()).isEmpty();
  }

  @Test
  void theProcessorSaysWhetherItWrites() {
    assertThat(new ManifestMigrationItemProcessor().applies()).isFalse();
    assertThat(new ManifestMigrationItemProcessor(false).applies()).isFalse();
    assertThat(new ManifestMigrationItemProcessor(true).applies()).isTrue();
  }

  @Test
  void theTaskletDefaultsToReportingWithoutWriting(@TempDir Path base) throws Exception {
    writeOldCollection(base, "a");
    byte[] before = Files.readAllBytes(manifestOf(base));

    // The single-argument constructor: a job that did not say "apply" must not write.
    new ManifestMigrationTasklet(base).execute(contribution(), chunkContext());

    assertThat(Files.readAllBytes(manifestOf(base)))
        .describedAs("the reporting tasklet must not rewrite anything")
        .isEqualTo(before);
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

  private static org.springframework.batch.core.StepContribution contribution() throws Exception {
    return new org.springframework.batch.core.StepContribution(
        org.springframework.batch.core.StepExecution.class
            .getDeclaredConstructor(String.class, org.springframework.batch.core.JobExecution.class)
            .newInstance("migrate", new org.springframework.batch.core.JobExecution(1L)));
  }

  private static org.springframework.batch.core.scope.context.ChunkContext chunkContext()
      throws Exception {
    var jobExecution = new org.springframework.batch.core.JobExecution(1L);
    var stepExecution =
        org.springframework.batch.core.StepExecution.class
            .getDeclaredConstructor(String.class, org.springframework.batch.core.JobExecution.class)
            .newInstance("migrate", jobExecution);
    var stepContext = new org.springframework.batch.core.scope.context.StepContext(stepExecution);
    return new org.springframework.batch.core.scope.context.ChunkContext(stepContext);
  }
}
