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
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;
import static org.assertj.core.api.Assertions.assertThatIOException;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.VectorCollectionConfig;
import com.integrallis.vectors.db.storage.GenerationDirectory.GenerationSource;
import com.integrallis.vectors.db.storage.GenerationDirectory.RecoveryResult;
import com.integrallis.vectors.db.storage.GenerationDirectory.WriteResult;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class GenerationDirectoryTest {

  // ---------------------------------------------------------------------------
  // Fixtures: a tiny "trivial" GenerationSource that writes 8 bytes into each
  // of the three data files and fsyncs them. GenerationDirectory does not
  // validate file contents (the per-file mappers do), so this is sufficient
  // to exercise the directory-write protocol end-to-end.
  //
  // The three PAYLOAD constants below are opaque 8-byte blobs, NOT real
  // vectors.bin / idmap.bin / metadata.bin wire formats. Production code
  // would reject them on open (the mapped stores parse fixed headers that
  // are longer than 8 bytes). These tests never call openGeneration on a
  // collection built from this source — they only exercise the write
  // protocol + CRC walk-back machinery in GenerationDirectory.recover.
  // ---------------------------------------------------------------------------

  private static final byte[] VECTORS_PAYLOAD = new byte[] {1, 2, 3, 4, 5, 6, 7, 8};
  private static final byte[] IDMAP_PAYLOAD = new byte[] {9, 10, 11, 12, 13, 14, 15, 16};
  private static final byte[] METADATA_PAYLOAD = new byte[] {17, 18, 19, 20, 21, 22, 23, 24};

  private static final VectorCollectionConfig CONFIG =
      new VectorCollectionConfig(
          16, SimilarityFunction.EUCLIDEAN, IndexType.FLAT, QuantizerKind.NONE, Integer.MAX_VALUE);

  private static GenerationSource trivialSource() {
    return new GenerationSource() {
      @Override
      public void writeVectors(Path destination) throws IOException {
        writeAndFsync(destination, VECTORS_PAYLOAD);
      }

      @Override
      public void writeIdmap(Path destination) throws IOException {
        writeAndFsync(destination, IDMAP_PAYLOAD);
      }

      @Override
      public void writeMetadata(Path destination) throws IOException {
        writeAndFsync(destination, METADATA_PAYLOAD);
      }
    };
  }

  private static GenerationSource emptySource() {
    return new GenerationSource() {
      @Override
      public void writeVectors(Path destination) throws IOException {
        writeAndFsync(destination, new byte[0]);
      }

      @Override
      public void writeIdmap(Path destination) throws IOException {
        writeAndFsync(destination, new byte[0]);
      }

      @Override
      public void writeMetadata(Path destination) throws IOException {
        writeAndFsync(destination, new byte[0]);
      }
    };
  }

  private static void writeAndFsync(Path file, byte[] payload) throws IOException {
    try (FileChannel ch =
        FileChannel.open(file, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      ByteBuffer buf = ByteBuffer.wrap(payload);
      while (buf.hasRemaining()) {
        ch.write(buf);
      }
      ch.force(true);
    }
  }

  private static Manifest sampleManifest(long generationNumber) {
    // Compute the real CRC32 of each payload so the manifest matches what the trivial source
    // writes. recover() verifies per-file CRCs in addition to the manifest self-CRC; a generation
    // whose manifest-declared CRC doesn't match the on-disk bytes is treated as corrupt and walks
    // back to an older generation. Tests that want to model corruption must explicitly mutate the
    // bytes AFTER writing.
    return Manifest.build(
        CONFIG,
        generationNumber,
        /* createdEpochMillis */ 1_700_000_000_000L,
        /* liveCount */ 0L,
        /* vectorsBinLength */ VECTORS_PAYLOAD.length,
        /* vectorsBinCrc32 */ Checksums.ofBytes(VECTORS_PAYLOAD),
        /* metadataBinLength */ METADATA_PAYLOAD.length,
        /* metadataBinCrc32 */ Checksums.ofBytes(METADATA_PAYLOAD),
        /* idmapBinLength */ IDMAP_PAYLOAD.length,
        /* idmapBinCrc32 */ Checksums.ofBytes(IDMAP_PAYLOAD),
        /* graphBinLength */ 0L,
        /* graphBinCrc32 */ 0L,
        /* quantizedBinLength */ 0L,
        /* quantizedBinCrc32 */ 0L);
  }

  private static Manifest emptyManifest(long generationNumber) {
    return Manifest.build(
        CONFIG,
        generationNumber,
        /* createdEpochMillis */ 1_700_000_000_000L,
        /* liveCount */ 0L,
        /* vectorsBinLength */ 0L,
        /* vectorsBinCrc32 */ 0L,
        /* metadataBinLength */ 0L,
        /* metadataBinCrc32 */ 0L,
        /* idmapBinLength */ 0L,
        /* idmapBinCrc32 */ 0L,
        /* graphBinLength */ 0L,
        /* graphBinCrc32 */ 0L,
        /* quantizedBinLength */ 0L,
        /* quantizedBinCrc32 */ 0L);
  }

  // ---------------------------------------------------------------------------
  // Happy path: writeGeneration publishes everything atomically; recover() opens
  // the same generation back without any rewrites.
  // ---------------------------------------------------------------------------

  @Nested
  class HappyPath {

    @Test
    void writeAndRecoverSameGeneration(@TempDir Path tmp) throws IOException {
      Manifest manifest = sampleManifest(0L);
      WriteResult written = GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), manifest);

      assertThat(written.generationNumber()).isEqualTo(0L);
      assertThat(written.generationDir()).isEqualTo(tmp.resolve("gen-0000000000000000"));
      assertThat(written.manifest()).isEqualTo(manifest);
      assertThat(Files.exists(written.generationDir())).isTrue();
      assertThat(Files.exists(written.generationDir().resolve("vectors.bin"))).isTrue();
      assertThat(Files.exists(written.generationDir().resolve("idmap.bin"))).isTrue();
      assertThat(Files.exists(written.generationDir().resolve("metadata.bin"))).isTrue();
      assertThat(Files.exists(written.generationDir().resolve("manifest.bin"))).isTrue();
      assertThat(Files.exists(tmp.resolve("CURRENT"))).isTrue();
      assertThat(Files.exists(tmp.resolve("CURRENT.tmp"))).isFalse();
      assertThat(Files.exists(tmp.resolve(".gen-0000000000000000.tmp"))).isFalse();

      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(0L);

      RecoveryResult recovered = GenerationDirectory.recover(tmp, null, null);
      assertThat(recovered.generationNumber()).isEqualTo(0L);
      assertThat(recovered.generationDir()).isEqualTo(written.generationDir());
      assertThat(recovered.manifest()).isEqualTo(manifest);
      assertThat(recovered.rewroteCurrent()).isFalse();
      assertThat(recovered.createdEmpty()).isFalse();
    }

    @Test
    void writeMultipleGenerationsRecoversNewest(@TempDir Path tmp) throws IOException {
      GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));
      GenerationDirectory.writeGeneration(tmp, 1L, trivialSource(), sampleManifest(1L));
      GenerationDirectory.writeGeneration(tmp, 2L, trivialSource(), sampleManifest(2L));

      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(2L);
      assertThat(Files.exists(tmp.resolve("gen-0000000000000000"))).isTrue();
      assertThat(Files.exists(tmp.resolve("gen-0000000000000001"))).isTrue();
      assertThat(Files.exists(tmp.resolve("gen-0000000000000002"))).isTrue();

      RecoveryResult r = GenerationDirectory.recover(tmp, null, null);
      assertThat(r.generationNumber()).isEqualTo(2L);
      assertThat(r.rewroteCurrent()).isFalse();
    }

    @Test
    void recoveryIsIdempotent(@TempDir Path tmp) throws IOException {
      GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));
      RecoveryResult first = GenerationDirectory.recover(tmp, null, null);
      RecoveryResult second = GenerationDirectory.recover(tmp, null, null);
      assertThat(first.generationNumber()).isEqualTo(second.generationNumber());
      assertThat(first.generationDir()).isEqualTo(second.generationDir());
      assertThat(second.rewroteCurrent()).isFalse();
      assertThat(second.createdEmpty()).isFalse();
    }
  }

  // ---------------------------------------------------------------------------
  // Crash recovery — partial writes left behind by simulated crashes.
  // ---------------------------------------------------------------------------

  @Nested
  class CrashRecovery {

    @Test
    void halfWrittenTmpDirCleanedOnOpen(@TempDir Path tmp) throws IOException {
      // First publish a real generation so the tmp dir is purely "leftover noise" and recover()
      // has a valid CURRENT to fall back on after sweeping.
      GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));

      // Simulate a half-written .gen-0000000000000001.tmp/ from a crashed commit. Populate it
      // with arbitrary files to prove deleteRecursively walks the contents, not just the dir.
      Path stale = tmp.resolve(".gen-0000000000000001.tmp");
      Files.createDirectory(stale);
      Files.write(stale.resolve("vectors.bin"), new byte[] {1, 2, 3});
      Files.write(stale.resolve("idmap.bin"), new byte[] {4, 5, 6});

      RecoveryResult r = GenerationDirectory.recover(tmp, null, null);
      assertThat(Files.exists(stale)).isFalse();
      // CURRENT was already valid → recovery did not rewrite it.
      assertThat(r.generationNumber()).isEqualTo(0L);
      assertThat(r.rewroteCurrent()).isFalse();
    }

    @Test
    void strictTmpDirMatchingPreservesUnrelatedDotFiles(@TempDir Path tmp) throws IOException {
      // The recovery sweep deletes only directories whose name matches the strict
      // ".gen-NNNNNNNNNNNNNNNN.tmp" pattern (16 zero-padded digits between prefix and suffix).
      // A coincidentally-named user file or directory must survive recovery.
      GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));
      Path bareDot = tmp.resolve(".gen-.tmp");
      Path nonNumeric = tmp.resolve(".gen-abcdefghijklmnop.tmp");
      Path tooShort = tmp.resolve(".gen-000000000000001.tmp"); // 15 digits, off by one
      Files.createDirectory(bareDot);
      Files.createDirectory(nonNumeric);
      Files.createDirectory(tooShort);

      GenerationDirectory.recover(tmp, null, null);

      assertThat(Files.exists(bareDot)).isTrue();
      assertThat(Files.exists(nonNumeric)).isTrue();
      assertThat(Files.exists(tooShort)).isTrue();
    }

    @Test
    void multipleStaleTmpDirsAllCleaned(@TempDir Path tmp) throws IOException {
      GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));
      Path stale1 = tmp.resolve(".gen-0000000000000001.tmp");
      Path stale2 = tmp.resolve(".gen-0000000000000002.tmp");
      Path stale3 = tmp.resolve(".gen-0000000000000099.tmp");
      Files.createDirectory(stale1);
      Files.createDirectory(stale2);
      Files.createDirectory(stale3);

      GenerationDirectory.recover(tmp, null, null);
      assertThat(Files.exists(stale1)).isFalse();
      assertThat(Files.exists(stale2)).isFalse();
      assertThat(Files.exists(stale3)).isFalse();
    }

    @Test
    void currentDotTmpCleanedOnOpen(@TempDir Path tmp) throws IOException {
      GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));
      Path currentTmp = tmp.resolve("CURRENT.tmp");
      Files.write(currentTmp, new byte[] {0, 0, 0, 0, 0, 0, 0, 0});

      RecoveryResult r = GenerationDirectory.recover(tmp, null, null);
      assertThat(Files.exists(currentTmp)).isFalse();
      assertThat(r.generationNumber()).isEqualTo(0L);
      assertThat(r.rewroteCurrent()).isFalse();
    }

    @Test
    void missingCurrentRecoversFromNewestValidGen(@TempDir Path tmp) throws IOException {
      GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));
      GenerationDirectory.writeGeneration(tmp, 1L, trivialSource(), sampleManifest(1L));
      GenerationDirectory.writeGeneration(tmp, 2L, trivialSource(), sampleManifest(2L));

      // Simulate a CURRENT that was lost between commit and fsync.
      Files.delete(tmp.resolve("CURRENT"));

      RecoveryResult r = GenerationDirectory.recover(tmp, null, null);
      assertThat(r.generationNumber()).isEqualTo(2L);
      assertThat(r.rewroteCurrent()).isTrue();
      assertThat(r.createdEmpty()).isFalse();
      // CURRENT was rewritten during recovery → readCurrent now returns 2.
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(2L);
    }

    @Test
    void corruptedNewestManifestFallsBackToPreviousGen(@TempDir Path tmp) throws IOException {
      GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));
      GenerationDirectory.writeGeneration(tmp, 1L, trivialSource(), sampleManifest(1L));

      // Corrupt the newest manifest by overwriting it with garbage. Drop CURRENT so the
      // recovery path is forced into the gen-scan branch (otherwise it would refuse to fall
      // back when CURRENT itself names a corrupted gen).
      Files.write(tmp.resolve("gen-0000000000000001/manifest.bin"), new byte[64]);
      Files.delete(tmp.resolve("CURRENT"));

      RecoveryResult r = GenerationDirectory.recover(tmp, null, null);
      assertThat(r.generationNumber()).isEqualTo(0L);
      assertThat(r.rewroteCurrent()).isTrue();
      assertThat(r.createdEmpty()).isFalse();
    }

    @Test
    void currentPointsAtCorruptedGenFallsBackToPrevious(@TempDir Path tmp) throws IOException {
      GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));
      GenerationDirectory.writeGeneration(tmp, 1L, trivialSource(), sampleManifest(1L));
      // CURRENT points at gen-1 here. Corrupt gen-1's manifest in place.
      Files.write(tmp.resolve("gen-0000000000000001/manifest.bin"), new byte[64]);

      RecoveryResult r = GenerationDirectory.recover(tmp, null, null);
      // Recovery's "happy path" branch fails (CURRENT target invalid), so it scans descending
      // and lands on gen-0, then rewrites CURRENT to 0.
      assertThat(r.generationNumber()).isEqualTo(0L);
      assertThat(r.rewroteCurrent()).isTrue();
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(0L);
    }

    @Test
    void currentPointsAtMissingGenFallsBack(@TempDir Path tmp) throws IOException {
      GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));
      // Hand-write CURRENT to point at a non-existent generation 99.
      writeCurrentBytes(tmp, 99L);

      RecoveryResult r = GenerationDirectory.recover(tmp, null, null);
      assertThat(r.generationNumber()).isEqualTo(0L);
      assertThat(r.rewroteCurrent()).isTrue();
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(0L);
    }

    @Test
    void allCorruptManifestsThrowsWithoutBootstrap(@TempDir Path tmp) throws IOException {
      GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));
      Files.write(tmp.resolve("gen-0000000000000000/manifest.bin"), new byte[64]);
      Files.delete(tmp.resolve("CURRENT"));

      assertThatIOException()
          .isThrownBy(() -> GenerationDirectory.recover(tmp, null, null))
          .withMessageContaining("no valid generation found");
    }

    @Test
    void corruptedPayloadInNewestGenFallsBackToPrevious(@TempDir Path tmp) throws IOException {
      // recover() verifies every payload file's CRC against the manifest-stored CRC, not just the
      // manifest self-CRC. A generation whose manifest is perfectly readable but whose vectors.bin
      // has been silently corrupted (bit rot, partial truncation, fsck-after-crash) is rejected and
      // the walk-back rolls forward to the newest gen whose payload bytes also match the manifest
      // CRCs.
      GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));
      GenerationDirectory.writeGeneration(tmp, 1L, trivialSource(), sampleManifest(1L));

      // Overwrite gen-1's vectors.bin with garbage of the same length. The manifest still
      // parses (self-CRC intact) but the stored vectors.bin CRC no longer matches the file.
      Path corruptVectors = tmp.resolve("gen-0000000000000001/vectors.bin");
      Files.write(corruptVectors, new byte[VECTORS_PAYLOAD.length]);

      RecoveryResult r = GenerationDirectory.recover(tmp, null, null);
      assertThat(r.generationNumber()).isEqualTo(0L);
      assertThat(r.rewroteCurrent()).isTrue();
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(0L);
    }

    @Test
    void currentPointsAtCorruptedPayloadFallsBackToPrevious(@TempDir Path tmp) throws IOException {
      // Same as above but CURRENT still names the corrupt generation (not deleted upfront).
      // The happy-path branch inside recover() must fail the per-file CRC check and fall
      // through to the descending scan rather than returning the corrupt generation.
      GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));
      GenerationDirectory.writeGeneration(tmp, 1L, trivialSource(), sampleManifest(1L));

      // Corrupt idmap.bin this time to prove it's not just vectors.bin that's checked.
      Path corruptIdmap = tmp.resolve("gen-0000000000000001/idmap.bin");
      Files.write(corruptIdmap, new byte[IDMAP_PAYLOAD.length]);

      RecoveryResult r = GenerationDirectory.recover(tmp, null, null);
      assertThat(r.generationNumber()).isEqualTo(0L);
      assertThat(r.rewroteCurrent()).isTrue();
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(0L);
    }

    @Test
    void payloadLengthMismatchFallsBackToPrevious(@TempDir Path tmp) throws IOException {
      // A payload that has been truncated (not just mutated) also fails the per-file check
      // via the length comparison inside verifyOneFile, BEFORE computing the CRC.
      GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));
      GenerationDirectory.writeGeneration(tmp, 1L, trivialSource(), sampleManifest(1L));

      Path truncatedMetadata = tmp.resolve("gen-0000000000000001/metadata.bin");
      Files.write(truncatedMetadata, new byte[METADATA_PAYLOAD.length - 2]);

      RecoveryResult r = GenerationDirectory.recover(tmp, null, null);
      assertThat(r.generationNumber()).isEqualTo(0L);
      assertThat(r.rewroteCurrent()).isTrue();
    }

    @Test
    void allCorruptManifestsBootstrapWhenSourceProvided(@TempDir Path tmp) throws IOException {
      // Note: even if bootstrapSource is provided, the bootstrap path will fail because gen-0
      // already exists on disk (writeGeneration refuses to clobber it). This test documents
      // that gotcha — recovery does not delete corrupted gen directories, only their tmp
      // siblings. Callers must surface this as an "unrecoverable" error.
      GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));
      Files.write(tmp.resolve("gen-0000000000000000/manifest.bin"), new byte[64]);
      Files.delete(tmp.resolve("CURRENT"));

      assertThatIOException()
          .isThrownBy(() -> GenerationDirectory.recover(tmp, emptySource(), emptyManifest(0L)))
          .withMessageContaining("generation directory already exists");
    }
  }

  // ---------------------------------------------------------------------------
  // Empty-directory bootstrap.
  // ---------------------------------------------------------------------------

  @Nested
  class EmptyDirectoryBootstrap {

    @Test
    void freshOpenWithBootstrapCreatesGenZero(@TempDir Path tmp) throws IOException {
      RecoveryResult r = GenerationDirectory.recover(tmp, emptySource(), emptyManifest(0L));
      assertThat(r.generationNumber()).isEqualTo(0L);
      assertThat(r.createdEmpty()).isTrue();
      assertThat(r.rewroteCurrent()).isTrue();
      assertThat(Files.exists(tmp.resolve("gen-0000000000000000"))).isTrue();
      assertThat(Files.exists(tmp.resolve("CURRENT"))).isTrue();
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(0L);
    }

    @Test
    void freshOpenWithoutBootstrapThrows(@TempDir Path tmp) {
      assertThatIOException()
          .isThrownBy(() -> GenerationDirectory.recover(tmp, null, null))
          .withMessageContaining("no valid generation found");
    }

    @Test
    void recoverCreatesMissingDirectory(@TempDir Path tmp) throws IOException {
      Path nested = tmp.resolve("collection").resolve("nested");
      assertThat(Files.exists(nested)).isFalse();
      GenerationDirectory.recover(nested, emptySource(), emptyManifest(0L));
      assertThat(Files.isDirectory(nested)).isTrue();
      assertThat(Files.exists(nested.resolve("CURRENT"))).isTrue();
    }

    @Test
    void rejectsBootstrapManifestWithNonZeroGeneration(@TempDir Path tmp) {
      assertThatIllegalArgumentException()
          .isThrownBy(() -> GenerationDirectory.recover(tmp, emptySource(), emptyManifest(7L)))
          .withMessageContaining("bootstrapManifest generationNumber must be 0");
    }

    @Test
    void rejectsBootstrapMismatch(@TempDir Path tmp) {
      assertThatIllegalArgumentException()
          .isThrownBy(() -> GenerationDirectory.recover(tmp, emptySource(), null))
          .withMessageContaining("must be either both null or both non-null");
      assertThatIllegalArgumentException()
          .isThrownBy(() -> GenerationDirectory.recover(tmp, null, emptyManifest(0L)))
          .withMessageContaining("must be either both null or both non-null");
    }
  }

  // ---------------------------------------------------------------------------
  // CURRENT pointer atomic update.
  // ---------------------------------------------------------------------------

  @Nested
  class WriteCurrentAtomic {

    @Test
    void writesEightBytesLittleEndian(@TempDir Path tmp) throws IOException {
      GenerationDirectory.writeCurrentAtomic(tmp, 0xCAFEBABEL);
      byte[] raw = Files.readAllBytes(tmp.resolve("CURRENT"));
      assertThat(raw).hasSize(8);
      long decoded = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).getLong();
      assertThat(decoded).isEqualTo(0xCAFEBABEL);
    }

    @Test
    void overwritesExistingCurrent(@TempDir Path tmp) throws IOException {
      GenerationDirectory.writeCurrentAtomic(tmp, 1L);
      GenerationDirectory.writeCurrentAtomic(tmp, 42L);
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(42L);
      assertThat(Files.exists(tmp.resolve("CURRENT.tmp"))).isFalse();
    }

    @Test
    void cleansLeftoverCurrentTmp(@TempDir Path tmp) throws IOException {
      Files.write(tmp.resolve("CURRENT.tmp"), new byte[] {0x55, 0x55, 0x55, 0x55, 0, 0, 0, 0});
      GenerationDirectory.writeCurrentAtomic(tmp, 7L);
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(7L);
      assertThat(Files.exists(tmp.resolve("CURRENT.tmp"))).isFalse();
    }

    @Test
    void rejectsNegativeGenerationNumber(@TempDir Path tmp) {
      assertThatIllegalArgumentException()
          .isThrownBy(() -> GenerationDirectory.writeCurrentAtomic(tmp, -1L))
          .withMessageContaining("out of range");
    }

    @Test
    void rejectsOverflowingGenerationNumber(@TempDir Path tmp) {
      assertThatIllegalArgumentException()
          .isThrownBy(
              () ->
                  GenerationDirectory.writeCurrentAtomic(
                      tmp, FileFormat.MAX_GENERATION_NUMBER + 1L))
          .withMessageContaining("out of range");
    }
  }

  // ---------------------------------------------------------------------------
  // CURRENT pointer read.
  // ---------------------------------------------------------------------------

  @Nested
  class ReadCurrent {

    @Test
    void returnsMinusOneOnMissing(@TempDir Path tmp) throws IOException {
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(-1L);
    }

    @Test
    void returnsMinusOneOnTooShort(@TempDir Path tmp) throws IOException {
      Files.write(tmp.resolve("CURRENT"), new byte[] {1, 2, 3});
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(-1L);
    }

    @Test
    void returnsMinusOneOnNegativeValue(@TempDir Path tmp) throws IOException {
      writeCurrentBytes(tmp, -42L);
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(-1L);
    }

    @Test
    void returnsValueOnValid(@TempDir Path tmp) throws IOException {
      writeCurrentBytes(tmp, 12345L);
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(12345L);
    }

    @Test
    void readsExtraBytesIgnored(@TempDir Path tmp) throws IOException {
      // CURRENT is supposed to be exactly 8 bytes, but a generous reader treats trailing bytes
      // as best-effort. Document the behaviour.
      ByteBuffer buf = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
      buf.putLong(99L);
      buf.putLong(0xDEADBEEFL); // garbage trailer
      Files.write(tmp.resolve("CURRENT"), buf.array());
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(99L);
    }
  }

  // ---------------------------------------------------------------------------
  // writeGeneration validation.
  // ---------------------------------------------------------------------------

  @Nested
  class WriteGenerationValidation {

    @Test
    void rejectsExistingGenDir(@TempDir Path tmp) throws IOException {
      Files.createDirectory(tmp.resolve("gen-0000000000000000"));
      assertThatIOException()
          .isThrownBy(
              () ->
                  GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L)))
          .withMessageContaining("generation directory already exists");
    }

    @Test
    void rejectsExistingTmpDir(@TempDir Path tmp) throws IOException {
      Files.createDirectory(tmp.resolve(".gen-0000000000000000.tmp"));
      assertThatIOException()
          .isThrownBy(
              () ->
                  GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L)))
          .withMessageContaining("in-flight tmp generation directory already exists");
    }

    @Test
    void rejectsManifestGenMismatch(@TempDir Path tmp) {
      Manifest mismatched = sampleManifest(7L);
      assertThatIllegalArgumentException()
          .isThrownBy(
              () -> GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), mismatched))
          .withMessageContaining("does not match generationNumber");
    }

    @Test
    void rejectsNonExistentRoot(@TempDir Path tmp) {
      Path missing = tmp.resolve("does-not-exist");
      assertThatIOException()
          .isThrownBy(
              () ->
                  GenerationDirectory.writeGeneration(
                      missing, 0L, trivialSource(), sampleManifest(0L)))
          .withMessageContaining("does not exist");
    }

    @Test
    void rejectsRootThatIsARegularFile(@TempDir Path tmp) throws IOException {
      Path file = tmp.resolve("not-a-dir");
      Files.write(file, new byte[] {1, 2, 3});
      assertThatIOException()
          .isThrownBy(
              () ->
                  GenerationDirectory.writeGeneration(
                      file, 0L, trivialSource(), sampleManifest(0L)))
          .withMessageContaining("not a directory");
    }

    @Test
    void rejectsGenSlotOccupiedByFile(@TempDir Path tmp) throws IOException {
      // A regular file at the gen-NNNN/ path is corrupt state — surface a clear error rather
      // than the opaque "FileAlreadyExistsException" that Files.createDirectory would raise.
      Files.write(tmp.resolve("gen-0000000000000000"), new byte[] {1, 2, 3});
      assertThatIOException()
          .isThrownBy(
              () ->
                  GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L)))
          .withMessageContaining("non-directory file");
    }

    @Test
    void cleansTmpDirOnSourceFailure(@TempDir Path tmp) {
      // GenerationSource that throws on the second callback. Verifies tmp dir is wiped clean
      // before the exception propagates and that a subsequent retry can use the same gen num.
      GenerationSource flaky =
          new GenerationSource() {
            @Override
            public void writeVectors(Path destination) throws IOException {
              writeAndFsync(destination, VECTORS_PAYLOAD);
            }

            @Override
            public void writeIdmap(Path destination) throws IOException {
              throw new IOException("simulated idmap write failure");
            }

            @Override
            public void writeMetadata(Path destination) {
              throw new AssertionError("should not reach metadata callback");
            }
          };

      assertThatExceptionOfType(IOException.class)
          .isThrownBy(() -> GenerationDirectory.writeGeneration(tmp, 0L, flaky, sampleManifest(0L)))
          .withMessageContaining("simulated idmap write failure");

      assertThat(Files.exists(tmp.resolve(".gen-0000000000000000.tmp"))).isFalse();
      assertThat(Files.exists(tmp.resolve("gen-0000000000000000"))).isFalse();
    }

    @Test
    void cleansTmpDirOnRuntimeFailure(@TempDir Path tmp) {
      GenerationSource bomb =
          new GenerationSource() {
            @Override
            public void writeVectors(Path destination) throws IOException {
              writeAndFsync(destination, VECTORS_PAYLOAD);
            }

            @Override
            public void writeIdmap(Path destination) {
              throw new IllegalStateException("kaboom");
            }

            @Override
            public void writeMetadata(Path destination) {
              throw new AssertionError("should not reach metadata callback");
            }
          };

      assertThatExceptionOfType(IllegalStateException.class)
          .isThrownBy(() -> GenerationDirectory.writeGeneration(tmp, 0L, bomb, sampleManifest(0L)))
          .withMessageContaining("kaboom");

      assertThat(Files.exists(tmp.resolve(".gen-0000000000000000.tmp"))).isFalse();
    }

    @Test
    void retryAfterCleanupSucceeds(@TempDir Path tmp) throws IOException {
      GenerationSource flaky =
          new GenerationSource() {
            @Override
            public void writeVectors(Path destination) throws IOException {
              writeAndFsync(destination, VECTORS_PAYLOAD);
            }

            @Override
            public void writeIdmap(Path destination) throws IOException {
              throw new IOException("first attempt fails");
            }

            @Override
            public void writeMetadata(Path destination) {
              throw new AssertionError();
            }
          };
      assertThatIOException()
          .isThrownBy(
              () -> GenerationDirectory.writeGeneration(tmp, 0L, flaky, sampleManifest(0L)));
      // Same generation number on retry — must succeed because the tmp dir was cleaned.
      WriteResult written =
          GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));
      assertThat(written.generationNumber()).isEqualTo(0L);
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(0L);
    }
  }

  // ---------------------------------------------------------------------------
  // I.7: the payload files are written concurrently. These tests pin that the
  // full (graph + quantized + tombstones) fan-out lands every file durably and
  // recovers, and that a failure in any one concurrent writer still cleans the
  // tmp dir, leaves no gen-NNNN/, and never advances CURRENT.
  // ---------------------------------------------------------------------------

  @Nested
  class ConcurrentWriteProtocol {

    private static final byte[] GRAPH_PAYLOAD = new byte[] {31, 32, 33, 34, 35, 36, 37, 38};
    private static final byte[] QUANTIZED_PAYLOAD = new byte[] {41, 42, 43, 44, 45, 46, 47, 48};
    private static final byte[] TOMBSTONES_PAYLOAD = new byte[] {51, 52, 53, 54, 55, 56, 57, 58};

    // A graph index config so the manifest's indexType is in GRAPH_INDEX_TYPES and the graph file
    // is written. Quantizer stays NONE in the config (these tests exercise only the directory
    // write/recover/CRC machinery, never openGeneration), but the manifest still declares a
    // non-zero quantized payload so writeGeneration materializes quantized.bin.
    private final VectorCollectionConfig graphConfig =
        new VectorCollectionConfig(
            16,
            SimilarityFunction.EUCLIDEAN,
            IndexType.HNSW,
            QuantizerKind.NONE,
            Integer.MAX_VALUE,
            null,
            new VectorCollectionConfig.HnswParams(16, 200, 1),
            null);

    private GenerationSource fullSource() {
      return new GenerationSource() {
        @Override
        public void writeVectors(Path destination) throws IOException {
          writeAndFsync(destination, VECTORS_PAYLOAD);
        }

        @Override
        public void writeIdmap(Path destination) throws IOException {
          writeAndFsync(destination, IDMAP_PAYLOAD);
        }

        @Override
        public void writeMetadata(Path destination) throws IOException {
          writeAndFsync(destination, METADATA_PAYLOAD);
        }

        @Override
        public void writeGraph(Path destination) throws IOException {
          writeAndFsync(destination, GRAPH_PAYLOAD);
        }

        @Override
        public void writeQuantized(Path destination) throws IOException {
          writeAndFsync(destination, QUANTIZED_PAYLOAD);
        }

        @Override
        public void writeTombstones(Path destination) throws IOException {
          writeAndFsync(destination, TOMBSTONES_PAYLOAD);
        }
      };
    }

    private Manifest fullManifest(long generationNumber) {
      return Manifest.buildWithTombstones(
          graphConfig,
          generationNumber,
          /* liveCount */ 1L,
          /* vectorsBinLength */ VECTORS_PAYLOAD.length,
          /* vectorsBinCrc32 */ Checksums.ofBytes(VECTORS_PAYLOAD),
          /* metadataBinLength */ METADATA_PAYLOAD.length,
          /* metadataBinCrc32 */ Checksums.ofBytes(METADATA_PAYLOAD),
          /* idmapBinLength */ IDMAP_PAYLOAD.length,
          /* idmapBinCrc32 */ Checksums.ofBytes(IDMAP_PAYLOAD),
          /* graphBinLength */ GRAPH_PAYLOAD.length,
          /* graphBinCrc32 */ Checksums.ofBytes(GRAPH_PAYLOAD),
          /* quantizedBinLength */ QUANTIZED_PAYLOAD.length,
          /* quantizedBinCrc32 */ Checksums.ofBytes(QUANTIZED_PAYLOAD),
          /* tombstoneCount */ 1L,
          /* tombstonesBinLength */ TOMBSTONES_PAYLOAD.length,
          /* tombstonesBinCrc32 */ Checksums.ofBytes(TOMBSTONES_PAYLOAD));
    }

    @Test
    void writesEveryPayloadFileConcurrentlyAndRecovers(@TempDir Path tmp) throws IOException {
      WriteResult written =
          GenerationDirectory.writeGeneration(tmp, 0L, fullSource(), fullManifest(0L));

      Path genDir = written.generationDir();
      assertThat(genDir.resolve(FileFormat.VECTORS_FILE)).exists();
      assertThat(genDir.resolve(FileFormat.IDMAP_FILE)).exists();
      assertThat(genDir.resolve(FileFormat.METADATA_FILE)).exists();
      assertThat(genDir.resolve(FileFormat.GRAPH_FILE)).exists();
      assertThat(genDir.resolve(FileFormat.QUANTIZED_FILE)).exists();
      assertThat(genDir.resolve(FileFormat.TOMBSTONES_FILE)).exists();
      assertThat(genDir.resolve(FileFormat.MANIFEST_FILE)).exists();

      // recover() re-verifies the manifest self-CRC AND every declared payload CRC, so a green
      // recovery proves the concurrent fan-out wrote every byte correctly and durably.
      RecoveryResult recovered = GenerationDirectory.recover(tmp, null, null);
      assertThat(recovered.generationNumber()).isEqualTo(0L);
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(0L);
    }

    @Test
    void failureInOneConcurrentWriterCleansTmpAndDoesNotAdvanceCurrent(@TempDir Path tmp)
        throws IOException {
      // Land a good generation 0 first so CURRENT points at it.
      GenerationDirectory.writeGeneration(tmp, 0L, trivialSource(), sampleManifest(0L));
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(0L);

      // Generation 1 has one payload writer that throws while the others may already be writing.
      GenerationSource failing =
          new GenerationSource() {
            @Override
            public void writeVectors(Path destination) throws IOException {
              writeAndFsync(destination, VECTORS_PAYLOAD);
            }

            @Override
            public void writeIdmap(Path destination) throws IOException {
              writeAndFsync(destination, IDMAP_PAYLOAD);
            }

            @Override
            public void writeMetadata(Path destination) throws IOException {
              throw new IOException("simulated metadata write failure");
            }
          };

      assertThatIOException()
          .isThrownBy(
              () -> GenerationDirectory.writeGeneration(tmp, 1L, failing, sampleManifest(1L)))
          .withMessageContaining("simulated metadata write failure");

      // The half-written tmp dir is gone, no gen-1 was published, and CURRENT still points at 0.
      assertThat(tmp.resolve(FileFormat.generationTmpDirName(1L))).doesNotExist();
      assertThat(tmp.resolve(FileFormat.generationDirName(1L))).doesNotExist();
      assertThat(GenerationDirectory.readCurrent(tmp)).isEqualTo(0L);

      RecoveryResult recovered = GenerationDirectory.recover(tmp, null, null);
      assertThat(recovered.generationNumber()).isEqualTo(0L);
    }
  }

  // ---------------------------------------------------------------------------
  // Helper.
  // ---------------------------------------------------------------------------

  private static void writeCurrentBytes(Path root, long value) throws IOException {
    byte[] raw = new byte[8];
    ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).putLong(value);
    Files.write(root.resolve("CURRENT"), raw);
  }
}
