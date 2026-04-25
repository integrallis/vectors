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

/**
 * VCR operating modes that determine how LLM calls are handled during tests.
 *
 * <p>Inspired by the Python VCR implementation in maestro-langgraph, this enum provides flexible
 * options for recording and playing back LLM responses.
 */
public enum VCRMode {

  /**
   * Use cached responses only. Fails if no cassette exists for a call. This is the default mode for
   * CI/CD environments where API calls should not be made.
   */
  PLAYBACK,

  /**
   * Always make real API calls and overwrite any existing cassettes. Use this mode when
   * re-recording all tests.
   */
  RECORD,

  /**
   * Only record tests that are not already in the registry. Existing cassettes are played back, new
   * tests are recorded.
   */
  RECORD_NEW,

  /** Re-record only tests that previously failed. Successful tests use existing cassettes. */
  RECORD_FAILED,

  /**
   * Smart mode: use cache if it exists, otherwise record. Good for development when you want
   * automatic recording of new tests.
   */
  PLAYBACK_OR_RECORD,

  /**
   * Disable VCR entirely. All calls go to real APIs, nothing is cached. Use this mode when testing
   * real API behavior.
   */
  OFF;

  /**
   * Checks if this mode can potentially make real API calls and record responses.
   *
   * @return true if this mode can record new cassettes
   */
  public boolean isRecordMode() {
    return this == RECORD
        || this == RECORD_NEW
        || this == RECORD_FAILED
        || this == PLAYBACK_OR_RECORD;
  }

  /**
   * Checks if this mode can use cached responses.
   *
   * @return true if this mode can play back existing cassettes
   */
  public boolean isPlaybackMode() {
    return this == PLAYBACK || this == PLAYBACK_OR_RECORD;
  }
}
