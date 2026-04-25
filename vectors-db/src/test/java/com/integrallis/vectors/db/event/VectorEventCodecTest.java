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
package com.integrallis.vectors.db.event;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.MetadataValue;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class VectorEventCodecTest {

  private static float[] vec(float... values) {
    return values;
  }

  @Test
  void addEvent_roundTrip() {
    VectorEvent.Add original =
        new VectorEvent.Add(
            "doc-1",
            vec(1f, 2f, 3f),
            "hello",
            Map.of("score", MetadataValue.of(0.9), "active", MetadataValue.of(true)));

    VectorEvent decoded = VectorEventCodec.decode(VectorEventCodec.encode(original));

    assertThat(decoded).isInstanceOf(VectorEvent.Add.class);
    VectorEvent.Add add = (VectorEvent.Add) decoded;
    assertThat(add.id()).isEqualTo("doc-1");
    assertThat(add.vector()).containsExactly(1f, 2f, 3f);
    assertThat(add.text()).isEqualTo("hello");
    assertThat(add.metadata()).hasSize(2);
    assertThat(((MetadataValue.Num) add.metadata().get("score")).value()).isEqualTo(0.9);
    assertThat(((MetadataValue.Bool) add.metadata().get("active")).value()).isTrue();
  }

  @Test
  void deleteEvent_roundTrip() {
    VectorEvent.Delete original = new VectorEvent.Delete("to-remove");
    VectorEvent decoded = VectorEventCodec.decode(VectorEventCodec.encode(original));

    assertThat(decoded).isInstanceOf(VectorEvent.Delete.class);
    assertThat(((VectorEvent.Delete) decoded).id()).isEqualTo("to-remove");
  }

  @Test
  void upsertEvent_roundTrip() {
    VectorEvent.Upsert original = new VectorEvent.Upsert("doc-2", vec(4f, 5f), null, Map.of());

    VectorEvent decoded = VectorEventCodec.decode(VectorEventCodec.encode(original));

    assertThat(decoded).isInstanceOf(VectorEvent.Upsert.class);
    VectorEvent.Upsert upsert = (VectorEvent.Upsert) decoded;
    assertThat(upsert.id()).isEqualTo("doc-2");
    assertThat(upsert.vector()).containsExactly(4f, 5f);
    assertThat(upsert.text()).isNull();
  }

  @Test
  void commitEvent_roundTrip() {
    VectorEvent.Commit original = new VectorEvent.Commit(42L);
    VectorEvent decoded = VectorEventCodec.decode(VectorEventCodec.encode(original));

    assertThat(decoded).isInstanceOf(VectorEvent.Commit.class);
    assertThat(((VectorEvent.Commit) decoded).generationNumber()).isEqualTo(42L);
  }

  @Test
  void allMetadataVariants_roundTrip() {
    VectorEvent.Add original =
        new VectorEvent.Add(
            "meta-doc",
            vec(1f),
            null,
            Map.of(
                "str", MetadataValue.of("hello"),
                "num", MetadataValue.of(3.14),
                "bool", MetadataValue.of(false),
                "tags", MetadataValue.tags("a", "b", "c")));

    VectorEvent decoded = VectorEventCodec.decode(VectorEventCodec.encode(original));
    Map<String, MetadataValue> meta = ((VectorEvent.Add) decoded).metadata();

    assertThat(((MetadataValue.Str) meta.get("str")).value()).isEqualTo("hello");
    assertThat(((MetadataValue.Num) meta.get("num")).value()).isEqualTo(3.14);
    assertThat(((MetadataValue.Bool) meta.get("bool")).value()).isFalse();
    assertThat(((MetadataValue.Tags) meta.get("tags")).values())
        .containsExactlyInAnyOrder("a", "b", "c");
  }

  @Test
  void highDimensionVector_roundTrip() {
    float[] big = new float[1536];
    for (int i = 0; i < big.length; i++) big[i] = i * 0.001f;
    VectorEvent.Add original = new VectorEvent.Add("big", big, null, Map.of());

    VectorEvent decoded = VectorEventCodec.decode(VectorEventCodec.encode(original));
    assertThat(((VectorEvent.Add) decoded).vector()).isEqualTo(big);
  }

  @Test
  void invalidTagThrows() {
    assertThatThrownBy(() -> VectorEventCodec.decode(new byte[] {(byte) 99}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void encodeIsCompact_deleteEventIsSmallerThanAddEvent() {
    byte[] addBytes =
        VectorEventCodec.encode(new VectorEvent.Add("id", vec(1f, 2f), null, Map.of()));
    byte[] delBytes = VectorEventCodec.encode(new VectorEvent.Delete("id"));
    assertThat(delBytes.length).isLessThan(addBytes.length);
  }
}
