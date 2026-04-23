package com.integrallis.vectors.vcr.serde.jackson;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteSerializer;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class JacksonCassetteSerializerTest {

  private final CassetteSerializer serializer = new JacksonCassetteSerializer();

  @Test
  void roundTripEmbedding() {
    CassetteRecord.Embedding in =
        new CassetteRecord.Embedding("T:1", "m", 42L, new float[] {1f, -2.5f, 3e-3f});
    CassetteRecord out = serializer.deserialize(serializer.serialize(in));
    assertInstanceOf(CassetteRecord.Embedding.class, out);
    CassetteRecord.Embedding back = (CassetteRecord.Embedding) out;
    assertEquals("T:1", back.testId());
    assertEquals("m", back.model());
    assertEquals(42L, back.timestamp());
    assertArrayEquals(in.embedding(), back.embedding());
  }

  @Test
  void roundTripBatchEmbedding() {
    float[][] batch = {
      {1f, 2f},
      {3f, 4f, 5f}
    };
    CassetteRecord.BatchEmbedding in = new CassetteRecord.BatchEmbedding("T:b", "m", 0L, batch);
    CassetteRecord.BatchEmbedding back =
        (CassetteRecord.BatchEmbedding) serializer.deserialize(serializer.serialize(in));
    assertEquals(2, back.embeddings().length);
    assertArrayEquals(batch[0], back.embeddings()[0]);
    assertArrayEquals(batch[1], back.embeddings()[1]);
  }

  @Test
  void roundTripChat() {
    CassetteRecord.Chat in =
        new CassetteRecord.Chat(
            "T:c", "gpt", 5L, "hello", "world", Map.of("role", "assistant", "k", "v"));
    CassetteRecord.Chat back =
        (CassetteRecord.Chat) serializer.deserialize(serializer.serialize(in));
    assertEquals("hello", back.prompt());
    assertEquals("world", back.response());
    assertEquals("assistant", back.metadata().get("role"));
    assertEquals("v", back.metadata().get("k"));
  }

  @Test
  void contentTypeIsJson() {
    assertEquals("application/json", serializer.contentType());
  }

  @Test
  void serviceLoaderFindsSerializer() {
    CassetteSerializer loaded = CassetteSerializer.load();
    assertInstanceOf(JacksonCassetteSerializer.class, loaded);
  }
}
