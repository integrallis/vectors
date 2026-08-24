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

/** Raised when strict playback finds a cassette recorded for a different model request. */
public final class VCRCassetteStaleException extends RuntimeException {

  private static final long serialVersionUID = 1L;

  /**
   * Creates a stale-cassette failure.
   *
   * @param cassetteKey serialized cassette key
   * @param recordedSignature signature stored with the cassette, possibly {@code null}
   * @param currentSignature signature of the current call
   */
  public VCRCassetteStaleException(
      String cassetteKey, String recordedSignature, String currentSignature) {
    super(
        "Cassette "
            + cassetteKey
            + " does not match the current model request (recorded="
            + (recordedSignature == null ? "legacy/unsigned" : recordedSignature)
            + ", current="
            + currentSignature
            + "). Run with VCRMode.PLAYBACK_OR_RECORD to refresh stale interactions.");
  }
}
