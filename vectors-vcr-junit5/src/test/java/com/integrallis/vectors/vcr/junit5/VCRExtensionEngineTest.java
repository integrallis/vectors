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
package com.integrallis.vectors.vcr.junit5;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.storage.backend.LocalFileStorageBackend;
import com.integrallis.vectors.vcr.CassetteKey;
import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ExactCassetteStore;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.platform.engine.discovery.DiscoverySelectors;
import org.junit.platform.testkit.engine.EngineTestKit;
import org.junit.platform.testkit.engine.Events;

/**
 * Drives {@link VCRExtension} through {@code junit-platform-testkit}: records a cassette in {@code
 * RECORD} mode, then replays it in {@code PLAYBACK} mode and verifies the recorded vector is
 * returned.
 */
@Tag("unit")
class VCRExtensionEngineTest {

  /**
   * System-property key used by {@link RecordPlaybackScenario} to decide which inner model to
   * install.
   */
  static final String MODE_PROP = "vcr.test.mode";

  private Path dataDir;

  @BeforeEach
  void setUp() throws IOException {
    dataDir = Path.of(System.getProperty("java.io.tmpdir"), "vcr-junit5-" + UUID.randomUUID());
    Files.createDirectories(dataDir);
    System.setProperty(VCRExtension.DATA_DIR_SYSPROP, dataDir.toString());
  }

  @AfterEach
  void tearDown() throws IOException {
    System.clearProperty(VCRExtension.DATA_DIR_SYSPROP);
    System.clearProperty(MODE_PROP);
    if (Files.exists(dataDir)) {
      try (var s = Files.walk(dataDir)) {
        s.sorted(Comparator.reverseOrder())
            .forEach(
                p -> {
                  try {
                    Files.deleteIfExists(p);
                  } catch (IOException ignored) {
                    // best-effort
                  }
                });
      }
    }
  }

  @Test
  void recordsOnFirstRunAndReplaysOnSecond() throws IOException {
    System.setProperty(MODE_PROP, "RECORD");
    Events record =
        EngineTestKit.engine("junit-jupiter")
            .selectors(DiscoverySelectors.selectClass(RecordPlaybackScenario.class))
            .execute()
            .testEvents();
    record.assertStatistics(stats -> stats.succeeded(1).failed(0));

    ExactCassetteStore exact = new ExactCassetteStore(new LocalFileStorageBackend(dataDir));
    CassetteKey expected =
        new CassetteKey("embedding", RecordPlaybackScenario.class.getName() + ":embeds", 1);
    Optional<CassetteRecord> stored = exact.retrieve(expected);
    assertThat(stored).isPresent();
    CassetteRecord.Embedding emb = (CassetteRecord.Embedding) stored.get();
    assertThat(emb.embedding()).containsExactly(5f, 42f);

    System.setProperty(MODE_PROP, "PLAYBACK");
    Events playback =
        EngineTestKit.engine("junit-jupiter")
            .selectors(DiscoverySelectors.selectClass(RecordPlaybackScenario.class))
            .execute()
            .testEvents();
    playback.assertStatistics(stats -> stats.succeeded(1).failed(0));
  }

  @Test
  void flushFailureFailsTeardownAndStillClosesStore() {
    FailingCassetteStore store = new FailingCassetteStore(true, false);

    assertThatThrownBy(() -> VCRExtension.flushAndClose(store))
        .isInstanceOf(IOException.class)
        .hasMessage("flush failed");
    assertThat(store.closed).isTrue();
  }

  @Test
  void closeFailureIsSuppressedWhenFlushAlsoFails() {
    FailingCassetteStore store = new FailingCassetteStore(true, true);

    assertThatThrownBy(() -> VCRExtension.flushAndClose(store))
        .isInstanceOf(IOException.class)
        .hasMessage("flush failed")
        .satisfies(t -> assertThat(t.getSuppressed()).hasSize(1))
        .satisfies(t -> assertThat(t.getSuppressed()[0]).hasMessage("close failed"));
    assertThat(store.closed).isTrue();
  }

  private static final class FailingCassetteStore implements CassetteStore {
    private final boolean failFlush;
    private final boolean failClose;
    private boolean closed;

    private FailingCassetteStore(boolean failFlush, boolean failClose) {
      this.failFlush = failFlush;
      this.failClose = failClose;
    }

    @Override
    public void store(CassetteKey key, CassetteRecord record) {}

    @Override
    public Optional<CassetteRecord> retrieve(CassetteKey key) {
      return Optional.empty();
    }

    @Override
    public boolean exists(CassetteKey key) {
      return false;
    }

    @Override
    public void delete(CassetteKey key) {}

    @Override
    public List<CassetteKey> listByTestId(String testId) {
      return List.of();
    }

    @Override
    public void flush() throws IOException {
      if (failFlush) {
        throw new IOException("flush failed");
      }
    }

    @Override
    public void close() throws IOException {
      closed = true;
      if (failClose) {
        throw new IOException("close failed");
      }
    }
  }
}
