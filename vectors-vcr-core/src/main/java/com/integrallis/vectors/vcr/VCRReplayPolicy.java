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

import java.util.Objects;
import java.util.Optional;

/** Shared exact-cassette policy for signature-aware framework adapters. */
public final class VCRReplayPolicy {

  private VCRReplayPolicy() {}

  /**
   * Determines whether a wrapper should replay {@code existing}; {@code false} means call the live
   * delegate and overwrite the ordinal cassette.
   *
   * @param mode effective VCR mode
   * @param existing cassette at the current ordinal key
   * @param key current cassette key
   * @param currentSignature signature of the complete current request
   * @return {@code true} to replay, {@code false} to record
   */
  public static boolean shouldReplay(
      VCRMode mode,
      Optional<? extends CassetteRecord> existing,
      CassetteKey key,
      String currentSignature) {
    Objects.requireNonNull(mode, "mode");
    Objects.requireNonNull(existing, "existing");
    Objects.requireNonNull(key, "key");
    Objects.requireNonNull(currentSignature, "currentSignature");

    if (!mode.isPlaybackMode()) {
      return false;
    }
    if (existing.isEmpty()) {
      if (mode == VCRMode.PLAYBACK) {
        throw new VCRCassetteMissingException(key.serializedKey(), key.testId());
      }
      return false;
    }

    String recordedSignature = existing.get().requestSignature();
    if (currentSignature.equals(recordedSignature)) {
      return true;
    }
    if (mode == VCRMode.PLAYBACK_OR_RECORD) {
      return false;
    }
    throw new VCRCassetteStaleException(key.serializedKey(), recordedSignature, currentSignature);
  }
}
