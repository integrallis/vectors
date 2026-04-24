package com.integrallis.vectors.cache.semantic;

/**
 * Bidirectional codec between cache payload {@code V} and a String representation stored in a
 * {@link com.integrallis.vectors.db.MetadataValue.Str} on the cached document.
 *
 * <p>Simple and format-agnostic. Callers pick their own serialization (raw string, JSON, Base64 of
 * a binary protocol, Avro, etc.). For {@code String} payloads use {@link #identity()}.
 */
public interface PayloadCodec<V> {

  /** Returns an identity codec for {@code String} payloads. */
  static PayloadCodec<String> identity() {
    return new PayloadCodec<>() {
      @Override
      public String encode(String value) {
        return value;
      }

      @Override
      public String decode(String encoded) {
        return encoded;
      }
    };
  }

  String encode(V value);

  V decode(String encoded);
}
