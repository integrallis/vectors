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
package com.integrallis.vectors.demo.serverclient;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * CI gate for the server-client demo (audit T3.10). Runs the full create → upsert → search →
 * describe → drop HTTP round-trip and asserts the demo completes end-to-end without throwing.
 * Catches regressions in any of the HTTP routes the demo exercises (CollectionsRoutes,
 * DocumentsRoutes, SearchRoutes, AdminRoutes) plus the server boot/shutdown path itself.
 */
class ServerClientAppTest {

  @Test
  void httpRoundTripCompletesWithThreeHits() throws Exception {
    ServerClientApp.DemoResult r = ServerClientApp.runDemo();

    assertThat(r.serverPort()).as("server must have bound an ephemeral port").isPositive();
    assertThat(r.describeStatus()).as("describe must return 200 OK").isEqualTo(200);
    assertThat(r.searchHitIds())
        .as("k=3 search must return three hits over a five-doc corpus")
        .hasSize(3);
    // Hit ids must be drawn from the five seeded ids (no random fabrication).
    assertThat(r.searchHitIds()).allSatisfy(id -> assertThat(id).matches("doc-[1-5]"));
    assertThat(r.searchHitIds()).doesNotHaveDuplicates();
  }
}
