package com.integrallis.vectors.vcr.testng;

import static org.testng.Assert.assertEquals;

import com.integrallis.vectors.vcr.VCRMode;
import com.integrallis.vectors.vcr.VCRModel;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * Driven programmatically by {@link VCRListenerTest}. The class name intentionally omits {@code
 * Test} so it is excluded from Gradle's default test discovery.
 */
@VCRTestNG(mode = VCRMode.PLAYBACK_OR_RECORD, dataDir = "overridden-by-sysprop")
public class RecordPlaybackScenario {

  @VCRModel FakeEmbeddingModel embedder;

  @BeforeMethod
  public void setUp() {
    String mode = System.getProperty(VCRListenerTest.MODE_PROP);
    this.embedder =
        "PLAYBACK".equals(mode)
            ? prompt -> new float[] {-1f, -1f}
            : prompt -> new float[] {(float) prompt.length(), 42f};
  }

  @Test
  public void embeds() {
    float[] v = embedder.embed("hello");
    assertEquals(v[0], 5f);
    assertEquals(v[1], 42f);
  }
}
