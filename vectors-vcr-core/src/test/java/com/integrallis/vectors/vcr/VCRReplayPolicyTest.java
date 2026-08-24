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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class VCRReplayPolicyTest {

  private static final CassetteKey KEY = new CassetteKey("embedding", "Suite:test", 1);

  @Test
  void playbackOrRecordReplaysOnlyWhenTheRequestSignatureMatches() {
    CassetteRecord record = recording("sha256:current");

    assertThat(
            VCRReplayPolicy.shouldReplay(
                VCRMode.PLAYBACK_OR_RECORD, Optional.of(record), KEY, "sha256:current"))
        .isTrue();
    assertThat(
            VCRReplayPolicy.shouldReplay(
                VCRMode.PLAYBACK_OR_RECORD, Optional.of(record), KEY, "sha256:changed"))
        .isFalse();
  }

  @Test
  void playbackOrRecordRecordsMissingAndLegacyCassettes() {
    assertThat(
            VCRReplayPolicy.shouldReplay(
                VCRMode.PLAYBACK_OR_RECORD, Optional.empty(), KEY, "sha256:current"))
        .isFalse();
    assertThat(
            VCRReplayPolicy.shouldReplay(
                VCRMode.PLAYBACK_OR_RECORD, Optional.of(recording(null)), KEY, "sha256:current"))
        .isFalse();
  }

  @Test
  void strictPlaybackRejectsAStaleCassetteInsteadOfReturningTheWrongResponse() {
    assertThatThrownBy(
            () ->
                VCRReplayPolicy.shouldReplay(
                    VCRMode.PLAYBACK, Optional.of(recording("sha256:old")), KEY, "sha256:new"))
        .isInstanceOf(VCRCassetteStaleException.class)
        .hasMessageContaining("PLAYBACK_OR_RECORD");
  }

  @Test
  void playbackOrRecordCanBothReplayAndRecord() {
    assertThat(VCRMode.PLAYBACK_OR_RECORD.isPlaybackMode()).isTrue();
    assertThat(VCRMode.PLAYBACK_OR_RECORD.isRecordMode()).isTrue();
  }

  private static CassetteRecord recording(String signature) {
    return new CassetteRecord.Embedding("Suite:test", "model", 1L, new float[] {1f, 2f}, signature);
  }
}
