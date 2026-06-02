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
package com.integrallis.vectors.vcr.testng;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

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
import org.testng.TestNG;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * Exercises {@link VCRListener} end-to-end by programmatically driving TestNG against {@link
 * RecordPlaybackScenario} in {@code RECORD} and then {@code PLAYBACK} modes.
 */
public class VCRListenerTest {

  /** System-property key used by {@link RecordPlaybackScenario}. */
  public static final String MODE_PROP = "vcr.test.mode";

  private Path dataDir;

  @BeforeMethod(alwaysRun = true)
  public void setUp() throws IOException {
    dataDir = Path.of(System.getProperty("java.io.tmpdir"), "vcr-testng-" + UUID.randomUUID());
    Files.createDirectories(dataDir);
    System.setProperty(VCRListener.DATA_DIR_SYSPROP, dataDir.toString());
  }

  @AfterMethod(alwaysRun = true)
  public void tearDown() throws IOException {
    System.clearProperty(VCRListener.DATA_DIR_SYSPROP);
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

  @Test(groups = "unit")
  public void recordsOnFirstRunAndReplaysOnSecond() throws IOException {
    System.setProperty(MODE_PROP, "RECORD");
    runScenario();

    ExactCassetteStore exact = new ExactCassetteStore(new LocalFileStorageBackend(dataDir));
    CassetteKey expected =
        new CassetteKey("embedding", RecordPlaybackScenario.class.getName() + ":embeds", 1);
    Optional<CassetteRecord> stored = exact.retrieve(expected);
    assertTrue(stored.isPresent(), "cassette must have been recorded");
    CassetteRecord.Embedding emb = (CassetteRecord.Embedding) stored.get();
    assertEquals(emb.embedding()[0], 5f);
    assertEquals(emb.embedding()[1], 42f);

    System.setProperty(MODE_PROP, "PLAYBACK");
    runScenario();
  }

  @Test(groups = "unit")
  public void flushFailureFailsTeardownAndStillClosesStore() {
    FailingCassetteStore store = new FailingCassetteStore(true, false);

    IOException thrown = expectThrows(IOException.class, () -> VCRListener.flushAndClose(store));

    assertEquals(thrown.getMessage(), "flush failed");
    assertTrue(store.closed, "close must still be attempted after flush failure");
  }

  @Test(groups = "unit")
  public void closeFailureIsSuppressedWhenFlushAlsoFails() {
    FailingCassetteStore store = new FailingCassetteStore(true, true);

    IOException thrown = expectThrows(IOException.class, () -> VCRListener.flushAndClose(store));

    assertEquals(thrown.getMessage(), "flush failed");
    assertEquals(thrown.getSuppressed().length, 1);
    assertEquals(thrown.getSuppressed()[0].getMessage(), "close failed");
    assertTrue(store.closed, "close must still be attempted after flush failure");
  }

  private static void runScenario() {
    TestNG testng = new TestNG();
    testng.setTestClasses(new Class<?>[] {RecordPlaybackScenario.class});
    testng.setVerbose(0);
    testng.setUseDefaultListeners(false);
    testng.run();
    assertEquals(testng.getStatus(), 0, "scenario run must succeed");
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
