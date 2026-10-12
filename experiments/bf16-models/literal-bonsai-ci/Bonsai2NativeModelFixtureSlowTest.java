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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("slow")
class Bonsai2NativeModelFixtureSlowTest {
  @Test
  void releasedArtifactOracleAndFreshSessionSurviveNativeProjectionRouting() throws Exception {
    Path directory =
        Path.of(
            System.getProperty(
                "models.fixtures.directory",
                Path.of(System.getProperty("user.home"), ".jvllm", "models").toString()));
    Path model = directory.resolve("Ternary-Bonsai-2-27B-PTQ1_0.gguf");
    assertThat(Files.size(model)).isEqualTo(5_946_648_928L);
    var digest = MessageDigest.getInstance("SHA-256");
    try (var stream = new DigestInputStream(Files.newInputStream(model), digest)) {
      stream.transferTo(OutputStream.nullOutputStream());
    }
    assertThat(HexFormat.of().formatHex(digest.digest()))
        .isEqualTo("53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3");
    var configuration =
        new BackendConfiguration(
            Map.of(), Map.of(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "64"), List.of());
    try (var backend = RustFfmBackend.load(model, configuration)) {
      assertThat(backend.diagnostics().environment())
          .containsEntry("native-quantized-decode", "true");
      assertThat(backend.diagnostics().optimizations())
          .anySatisfy(
              optimization -> {
                assertThat(optimization.id()).isEqualTo("batched-prefill");
                assertThat(optimization.status().toString()).isEqualTo("ENABLED");
              });
      int[] tokens = backend.tokenizer().encode("The quick brown fox");
      assertThat(tokens).containsExactly(760, 3841, 13477, 37550);
      int[] expected = {33075, 888, 279, 15217, 5388, 13, 271, 760};
      float[] logits = backend.prefill(tokens, 0);
      for (int step = 0; step < expected.length; step++) {
        assertThat(argmax(logits)).as("Prism oracle token %d", step).isEqualTo(expected[step]);
        if (step + 1 < expected.length)
          logits = backend.forward(expected[step], tokens.length + step);
      }
      try (var session = backend.openSession()) {
        assertThat(argmax(backend.prefill(session, tokens, 0))).isEqualTo(expected[0]);
      }
      backend.reset();
      assertThat(argmax(backend.prefill(tokens, 0))).isEqualTo(expected[0]);
    }
  }

  @Test
  void transformedBatchesMatchSequentialRealWeightLogitsAcrossChunkBoundaries() throws Exception {
    Path model =
        Path.of(
                System.getProperty(
                    "models.fixtures.directory",
                    Path.of(System.getProperty("user.home"), ".jvllm", "models").toString()))
            .resolve("Ternary-Bonsai-2-27B-PTQ1_0.gguf");
    String prompt =
        "Explain why independent requests must not share recurrent state. "
            + "A second request should start with its own history and produce the same answer when repeated.";
    float[][] expected = new float[3][];
    int[] greedy = new int[3];
    int[] tokens;
    var sequential =
        new BackendConfiguration(
            Map.of(),
            Map.of(
                PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY,
                "96",
                "models.purejava.prefillBatchSize",
                "1"),
            List.of());
    try (var backend = RustFfmBackend.load(model, sequential)) {
      tokens = backend.tokenizer().encode(prompt);
      assertThat(tokens.length).isGreaterThan(16);
      float[] logits = backend.prefill(tokens, 0);
      for (int i = 0; i < 3; i++) {
        expected[i] = logits.clone();
        greedy[i] = argmax(logits);
        if (i < 2) logits = backend.forward(greedy[i], tokens.length + i);
      }
    }
    var batched =
        new BackendConfiguration(
            Map.of(),
            Map.of(
                PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY,
                "96",
                "models.purejava.prefillBatchSize",
                "16"),
            List.of());
    try (var backend = RustFfmBackend.load(model, batched)) {
      float[] logits = backend.prefill(tokens, 0);
      for (int i = 0; i < 3; i++) {
        double largestScaledError = 0;
        double largestAbsoluteError = 0;
        for (int token = 0; token < logits.length; token++) {
          assertThat(Float.isFinite(logits[token])).isTrue();
          double error = Math.abs((double) logits[token] - expected[i][token]);
          largestAbsoluteError = Math.max(largestAbsoluteError, error);
          largestScaledError =
              Math.max(largestScaledError, error / (1 + Math.abs(expected[i][token])));
        }
        System.out.printf(
            "Bonsai batch logit parity step=%d maxAbsolute=%g maxScaled=%g%n",
            i, largestAbsoluteError, largestScaledError);
        assertThat(argmax(logits)).as("greedy continuation %d", i).isEqualTo(greedy[i]);
        assertThat(largestAbsoluteError)
            .as("full-vocabulary parity at continuation %d", i)
            .isZero();
        if (i < 2) logits = backend.forward(greedy[i], tokens.length + i);
      }
    }
  }

  private static int argmax(float[] logits) {
    int best = 0;
    for (int i = 0; i < logits.length; i++) {
      assertThat(Float.isFinite(logits[i])).isTrue();
      if (logits[i] > logits[best]) best = i;
    }
    return best;
  }
}
