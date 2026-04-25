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

import com.integrallis.vectors.vcr.VCRMode;
import com.integrallis.vectors.vcr.VCRModel;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Scenario class driven by {@link VCRExtensionEngineTest} through {@code junit-platform-testkit}.
 * The class name intentionally does not match the default {@code *Test}/{@code Test*}/{@code
 * *Tests} patterns so Gradle's default test task does not pick it up directly.
 */
@VCRTest(mode = VCRMode.PLAYBACK_OR_RECORD, dataDir = "overridden-by-sysprop")
public class RecordPlaybackScenario {

  @VCRModel FakeEmbeddingModel embedder;

  @BeforeEach
  void setUp() {
    String mode = System.getProperty(VCRExtensionEngineTest.MODE_PROP);
    this.embedder =
        "PLAYBACK".equals(mode)
            ? prompt -> new float[] {-1f, -1f}
            : prompt -> new float[] {(float) prompt.length(), 42f};
  }

  @Test
  void embeds() {
    float[] v = embedder.embed("hello");
    assertThat(v[0]).isEqualTo(5f);
    assertThat(v[1]).isEqualTo(42f);
  }
}
