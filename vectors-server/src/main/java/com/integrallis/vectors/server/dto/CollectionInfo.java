package com.integrallis.vectors.server.dto;

import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.VectorCollectionConfig;
import java.time.Instant;

/**
 * Outbound body for {@code GET /v1/collections/{name}} and {@code POST /v1/collections}.
 *
 * <p>Enum-valued fields are serialized as strings (e.g. {@code "COSINE"}, {@code "HNSW"}, {@code
 * "NONE"}) so the wire format stays stable across Java enum refactors.
 *
 * @param name URL-safe collection name
 * @param dimension fixed vector dimension
 * @param metric similarity function (e.g. {@code COSINE})
 * @param indexType index backend (e.g. {@code HNSW})
 * @param quantizer quantizer kind (e.g. {@code NONE})
 * @param size number of live documents currently committed
 * @param createdAt UTC creation instant (ISO-8601 via {@code JavaTimeModule})
 */
public record CollectionInfo(
    String name,
    int dimension,
    String metric,
    String indexType,
    String quantizer,
    int size,
    Instant createdAt) {

  /**
   * Projects a live {@link VectorCollection} plus a registry-tracked {@code createdAt} into the
   * outbound wire form.
   */
  public static CollectionInfo of(String name, VectorCollection collection, Instant createdAt) {
    VectorCollectionConfig config = collection.config();
    return new CollectionInfo(
        name,
        config.dimension(),
        config.metric().name(),
        config.indexType().name(),
        config.quantizerKind().name(),
        collection.size(),
        createdAt);
  }
}
