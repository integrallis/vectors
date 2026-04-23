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
