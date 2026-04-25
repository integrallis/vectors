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
package com.integrallis.vectors.vcr;

import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Tracks recording status for each test.
 *
 * <p>Status is persisted under the {@code vcr:registry:test:<testId>} key in the supplied {@link
 * StorageBackend}, and mirrored in an in-memory cache. Status values are serialized as raw ASCII
 * enum names so the registry does not depend on the JSON serialization SPI.
 */
public final class VCRRegistry {

  /** Status of a recorded test. */
  public enum RecordingStatus {
    /** The test has been successfully recorded. */
    RECORDED,
    /** The last recording attempt for this test failed. */
    FAILED,
    /** The test has no recording yet. */
    MISSING
  }

  private static final String REGISTRY_KEY_PREFIX = "vcr:registry:test:";

  private final StorageBackend backend;
  private final ConcurrentMap<String, RecordingStatus> cache = new ConcurrentHashMap<>();

  /**
   * Creates a registry persisting to {@code backend}. Pass a {@link
   * com.integrallis.vectors.storage.backend.HeapStorageBackend} for pure in-memory operation.
   *
   * @param backend the storage backend (must not be null)
   */
  public VCRRegistry(StorageBackend backend) {
    this.backend = Objects.requireNonNull(backend, "backend");
  }

  /**
   * Marks a test as successfully recorded.
   *
   * @param testId the test identifier
   */
  public void registerSuccess(String testId) {
    writeStatus(testId, RecordingStatus.RECORDED);
  }

  /**
   * Marks a test's recording attempt as failed.
   *
   * @param testId the test identifier
   */
  public void registerFailure(String testId) {
    writeStatus(testId, RecordingStatus.FAILED);
  }

  /**
   * Returns the current recording status of the given test.
   *
   * @param testId the test identifier
   * @return the status (never null; {@link RecordingStatus#MISSING} if unknown)
   */
  public RecordingStatus getTestStatus(String testId) {
    RecordingStatus cached = cache.get(testId);
    if (cached != null) {
      return cached;
    }
    try {
      byte[] bytes = backend.get(REGISTRY_KEY_PREFIX + testId);
      if (bytes == null) {
        return RecordingStatus.MISSING;
      }
      RecordingStatus status =
          RecordingStatus.valueOf(new String(bytes, StandardCharsets.US_ASCII));
      cache.put(testId, status);
      return status;
    } catch (IOException e) {
      throw new UncheckedIOException("VCR registry read failed for " + testId, e);
    }
  }

  /**
   * Resolves the effective mode for {@code testId} given the configured global mode.
   *
   * @param testId the test identifier
   * @param globalMode the globally configured mode
   * @return the effective mode (never null)
   */
  public VCRMode determineEffectiveMode(String testId, VCRMode globalMode) {
    RecordingStatus status = getTestStatus(testId);
    return switch (globalMode) {
      case RECORD_NEW -> status == RecordingStatus.MISSING ? VCRMode.RECORD : VCRMode.PLAYBACK;
      case RECORD_FAILED ->
          (status == RecordingStatus.FAILED || status == RecordingStatus.MISSING)
              ? VCRMode.RECORD
              : VCRMode.PLAYBACK;
      case PLAYBACK_OR_RECORD ->
          status == RecordingStatus.RECORDED ? VCRMode.PLAYBACK : VCRMode.RECORD;
      default -> globalMode;
    };
  }

  private void writeStatus(String testId, RecordingStatus status) {
    Objects.requireNonNull(testId, "testId");
    try {
      backend.put(REGISTRY_KEY_PREFIX + testId, status.name().getBytes(StandardCharsets.US_ASCII));
      cache.put(testId, status);
    } catch (IOException e) {
      throw new UncheckedIOException("VCR registry write failed for " + testId, e);
    }
  }
}
