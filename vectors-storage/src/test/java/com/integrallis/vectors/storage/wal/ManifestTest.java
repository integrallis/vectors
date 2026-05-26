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
package com.integrallis.vectors.storage.wal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIOException;

import java.io.IOException;
import java.util.List;
import org.junit.jupiter.api.Test;

class ManifestTest {

  @Test
  void toJsonRoundTripsAllManifestFields() throws IOException {
    Manifest manifest =
        new Manifest(
            42,
            3,
            29,
            List.of(
                new BackendWriteAheadLog.SegmentMeta(3, 30, 39, true),
                new BackendWriteAheadLog.SegmentMeta(4, 40, 42, false)));

    Manifest parsed = Manifest.parse(manifest.toJson());

    assertThat(parsed.lastSeq).isEqualTo(42);
    assertThat(parsed.compactedSegmentIndex).isEqualTo(3);
    assertThat(parsed.compactedThroughSeq).isEqualTo(29);
    assertThat(parsed.segments).hasSize(2);
    assertSegment(parsed.segments.get(0), 3, 30, 39, true);
    assertSegment(parsed.segments.get(1), 4, 40, 42, false);
  }

  @Test
  void parseSkipsUnknownNestedValuesAndWhitespace() throws IOException {
    String json =
        """
        {
          "ignored": {"array": [1, true, {"s": "x\\\"y"}]},
          "lastSeq": 7,
          "compactedSegmentIndex": 1,
          "compactedThroughSeq": 4,
          "segments": [
            {"index": 1, "firstSeq": 5, "lastSeq": 7, "indexed": false, "extra": [1, 2]}
          ]
        }
        """;

    Manifest parsed = Manifest.parse(json);

    assertThat(parsed.lastSeq).isEqualTo(7);
    assertThat(parsed.compactedSegmentIndex).isEqualTo(1);
    assertThat(parsed.compactedThroughSeq).isEqualTo(4);
    assertThat(parsed.segments).singleElement().satisfies(s -> assertSegment(s, 1, 5, 7, false));
  }

  @Test
  void parseRejectsMalformedManifestAsIOException() {
    assertThatIOException()
        .isThrownBy(() -> Manifest.parse("{\"lastSeq\":1,\"segments\":[}"))
        .withMessageContaining("malformed WAL manifest");
  }

  @Test
  void cursorRejectsInvalidNumber() {
    assertThatIOException()
        .isThrownBy(() -> Manifest.parse("{\"lastSeq\":,\"segments\":[]}"))
        .withMessageContaining("expected number");
  }

  private static void assertSegment(
      BackendWriteAheadLog.SegmentMeta segment,
      int index,
      long firstSeq,
      long lastSeq,
      boolean indexed) {
    assertThat(segment.index).isEqualTo(index);
    assertThat(segment.firstSeq).isEqualTo(firstSeq);
    assertThat(segment.lastSeq).isEqualTo(lastSeq);
    assertThat(segment.indexed).isEqualTo(indexed);
  }
}
