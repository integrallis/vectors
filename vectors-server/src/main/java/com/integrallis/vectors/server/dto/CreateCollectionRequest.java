package com.integrallis.vectors.server.dto;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.VectorCollectionBuilder;
import java.nio.file.Path;
import java.util.regex.Pattern;

/**
 * Inbound body for {@code POST /v1/collections}.
 *
 * <p>Wire format (all fields except {@code quantizer}, {@code hnswM}, {@code hnswEfConstruction},
 * and {@code autoCommitThreshold} are required):
 *
 * <pre>{@code
 * {
 *   "name":                 "articles",
 *   "dimension":            384,
 *   "metric":               "COSINE",
 *   "indexType":            "HNSW",
 *   "quantizer":            "NONE",
 *   "hnswM":                16,
 *   "hnswEfConstruction":   100,
 *   "autoCommitThreshold":  32
 * }
 * }</pre>
 *
 * <p>Enum strings are matched case-insensitively so clients may use either {@code cosine} or {@code
 * COSINE}. Unknown enum values produce a {@code 400 Bad Request}.
 */
public record CreateCollectionRequest(
    String name,
    Integer dimension,
    String metric,
    String indexType,
    String quantizer,
    Integer hnswM,
    Integer hnswEfConstruction,
    Integer autoCommitThreshold) {

  /** URL-safe name charset: letters, digits, hyphen, underscore. 1..128 characters. */
  private static final Pattern NAME_PATTERN = Pattern.compile("^[A-Za-z0-9_-]{1,128}$");

  @JsonCreator
  public CreateCollectionRequest(
      @JsonProperty("name") String name,
      @JsonProperty("dimension") Integer dimension,
      @JsonProperty("metric") String metric,
      @JsonProperty("indexType") String indexType,
      @JsonProperty("quantizer") String quantizer,
      @JsonProperty("hnswM") Integer hnswM,
      @JsonProperty("hnswEfConstruction") Integer hnswEfConstruction,
      @JsonProperty("autoCommitThreshold") Integer autoCommitThreshold) {
    this.name = name;
    this.dimension = dimension;
    this.metric = metric;
    this.indexType = indexType;
    this.quantizer = quantizer;
    this.hnswM = hnswM;
    this.hnswEfConstruction = hnswEfConstruction;
    this.autoCommitThreshold = autoCommitThreshold;
  }

  /**
   * Validates required fields and parses enum strings. Returns the first field-level error message
   * encountered, or {@code null} if the request is well-formed.
   */
  public String validate() {
    if (name == null || !NAME_PATTERN.matcher(name).matches()) {
      return "name must match " + NAME_PATTERN.pattern();
    }
    if (dimension == null || dimension <= 0) {
      return "dimension must be a positive integer";
    }
    if (metric == null) {
      return "metric is required (EUCLIDEAN|DOT_PRODUCT|COSINE|MAXIMUM_INNER_PRODUCT)";
    }
    try {
      SimilarityFunction.valueOf(metric.toUpperCase());
    } catch (IllegalArgumentException e) {
      return "unknown metric: " + metric;
    }
    if (indexType == null) {
      return "indexType is required (FLAT|HNSW|VAMANA|IVF_FLAT|IVF_PQ)";
    }
    try {
      IndexType.valueOf(indexType.toUpperCase());
    } catch (IllegalArgumentException e) {
      return "unknown indexType: " + indexType;
    }
    if (quantizer != null) {
      try {
        QuantizerKind.valueOf(quantizer.toUpperCase());
      } catch (IllegalArgumentException e) {
        return "unknown quantizer: " + quantizer;
      }
    }
    if (hnswM != null && hnswM <= 0) {
      return "hnswM must be positive";
    }
    if (hnswEfConstruction != null && hnswEfConstruction <= 0) {
      return "hnswEfConstruction must be positive";
    }
    if (autoCommitThreshold != null && autoCommitThreshold <= 0) {
      return "autoCommitThreshold must be positive";
    }
    return null;
  }

  /**
   * Builds a {@link VectorCollection} from this request under the given storage root.
   *
   * @param storageRoot optional absolute storage root; if {@code null} the collection is in-memory
   */
  public VectorCollection toCollection(Path storageRoot) {
    VectorCollectionBuilder builder =
        VectorCollection.builder()
            .dimension(dimension)
            .metric(SimilarityFunction.valueOf(metric.toUpperCase()))
            .indexType(IndexType.valueOf(indexType.toUpperCase()));
    if (quantizer != null) {
      builder.quantizer(QuantizerKind.valueOf(quantizer.toUpperCase()));
    }
    if (hnswM != null) {
      builder.hnswM(hnswM);
    }
    if (hnswEfConstruction != null) {
      builder.hnswEfConstruction(hnswEfConstruction);
    }
    if (autoCommitThreshold != null) {
      builder.autoCommitThreshold(autoCommitThreshold);
    }
    if (storageRoot != null) {
      builder.storagePath(storageRoot);
    }
    return builder.build();
  }
}
