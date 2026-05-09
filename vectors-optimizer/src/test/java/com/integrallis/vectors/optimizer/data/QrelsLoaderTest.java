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
package com.integrallis.vectors.optimizer.data;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class QrelsLoaderTest {

  @Test
  void loadsTrecStyleJson(@TempDir Path tmp) throws Exception {
    Path file = tmp.resolve("qrels.json");
    Files.writeString(file, "{\"q1\":{\"d1\":3,\"d2\":1},\"q2\":{\"d3\":2}}");

    Qrels qrels = Qrels.loadJson(file);
    assertThat(qrels.relevance()).hasSize(2);
    assertThat(qrels.relevance().get("q1")).containsEntry("d1", 3).containsEntry("d2", 1);
    assertThat(qrels.size()).isEqualTo(3);
  }

  @Test
  void rejectsNegativeRelevance() {
    assertThatThrownBy(() -> new Qrels(Map.of("q1", Map.of("d1", -1))))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Negative");
  }
}
