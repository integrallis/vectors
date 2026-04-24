package com.integrallis.vectors.vcr.serde.avaje;

import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteSerializer;
import io.avaje.jsonb.JsonType;
import io.avaje.jsonb.Jsonb;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@link CassetteSerializer} backed by Avaje {@code Jsonb}.
 *
 * <p>Serializes each {@link CassetteRecord} to a plain {@code Map} tree so no annotation processor
 * is required. The JSON output shape matches the Jackson implementation in {@code
 * vectors-vcr-serde-jackson} for cross-serializer interoperability.
 */
public final class AvajeCassetteSerializer implements CassetteSerializer {

  private static final String TYPE_EMBEDDING = "embedding";
  private static final String TYPE_BATCH_EMBEDDING = "batch_embedding";
  private static final String TYPE_CHAT = "chat";

  private final Jsonb jsonb = Jsonb.builder().build();
  private final JsonType<Object> anyType = jsonb.type(Object.class);

  @Override
  public byte[] serialize(CassetteRecord record) {
    Map<String, Object> tree = toTree(record);
    return anyType.toJsonBytes(tree);
  }

  @Override
  public CassetteRecord deserialize(byte[] bytes) {
    Object parsed = anyType.fromJson(bytes);
    if (!(parsed instanceof Map<?, ?> map)) {
      throw new IllegalArgumentException("expected JSON object at top level");
    }
    return fromTree(map);
  }

  private static Map<String, Object> toTree(CassetteRecord record) {
    Map<String, Object> tree = new LinkedHashMap<>();
    if (record instanceof CassetteRecord.Embedding e) {
      tree.put("type", TYPE_EMBEDDING);
      tree.put("testId", e.testId());
      tree.put("model", e.model());
      tree.put("timestamp", e.timestamp());
      tree.put("embedding", asList(e.embedding()));
    } else if (record instanceof CassetteRecord.BatchEmbedding b) {
      tree.put("type", TYPE_BATCH_EMBEDDING);
      tree.put("testId", b.testId());
      tree.put("model", b.model());
      tree.put("timestamp", b.timestamp());
      List<List<Double>> outer = new ArrayList<>(b.embeddings().length);
      for (float[] v : b.embeddings()) {
        outer.add(asList(v));
      }
      tree.put("embeddings", outer);
    } else if (record instanceof CassetteRecord.Chat c) {
      tree.put("type", TYPE_CHAT);
      tree.put("testId", c.testId());
      tree.put("model", c.model());
      tree.put("timestamp", c.timestamp());
      tree.put("prompt", c.prompt());
      tree.put("response", c.response());
      tree.put("metadata", new LinkedHashMap<>(c.metadata()));
    } else {
      throw new IllegalArgumentException("unsupported record type: " + record.getClass());
    }
    return tree;
  }

  private static CassetteRecord fromTree(Map<?, ?> map) {
    String type = (String) map.get("type");
    String testId = (String) map.get("testId");
    String model = (String) map.get("model");
    long timestamp = ((Number) map.get("timestamp")).longValue();
    return switch (type) {
      case TYPE_EMBEDDING ->
          new CassetteRecord.Embedding(
              testId, model, timestamp, toFloatArray((List<?>) map.get("embedding")));
      case TYPE_BATCH_EMBEDDING -> {
        List<?> outer = (List<?>) map.get("embeddings");
        float[][] embeddings = new float[outer.size()][];
        for (int i = 0; i < outer.size(); i++) {
          embeddings[i] = toFloatArray((List<?>) outer.get(i));
        }
        yield new CassetteRecord.BatchEmbedding(testId, model, timestamp, embeddings);
      }
      case TYPE_CHAT -> {
        Object raw = map.get("metadata");
        Map<String, String> metadata = new LinkedHashMap<>();
        if (raw instanceof Map<?, ?> m) {
          for (Map.Entry<?, ?> e : m.entrySet()) {
            metadata.put(String.valueOf(e.getKey()), String.valueOf(e.getValue()));
          }
        }
        String prompt = map.get("prompt") instanceof String p ? p : "";
        String response = (String) map.get("response");
        yield new CassetteRecord.Chat(testId, model, timestamp, prompt, response, metadata);
      }
      default -> throw new IllegalArgumentException("unknown cassette type: " + type);
    };
  }

  private static List<Double> asList(float[] values) {
    List<Double> out = new ArrayList<>(values.length);
    for (float v : values) {
      out.add((double) v);
    }
    return out;
  }

  private static float[] toFloatArray(List<?> list) {
    float[] arr = new float[list.size()];
    for (int i = 0; i < list.size(); i++) {
      arr[i] = ((Number) list.get(i)).floatValue();
    }
    return arr;
  }
}
