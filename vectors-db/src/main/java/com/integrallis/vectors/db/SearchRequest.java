package com.integrallis.vectors.db;

import com.integrallis.vectors.db.filter.Filter;
import java.util.Objects;

/**
 * A search request. Immutable; use the {@link Builder} for fluent construction with defaults.
 *
 * <p>Defaults applied by {@link Builder#build()}:
 *
 * <ul>
 *   <li>{@code searchListSize} = {@code max(k, 100)}
 *   <li>{@code overQueryFactor} = 4.0f
 *   <li>{@code minScore} = {@code -Float.MAX_VALUE} (no minimum)
 *   <li>{@code filter} = null (no filter; semantically equivalent to {@link
 *       com.integrallis.vectors.db.filter.Filters#all()})
 *   <li>{@code includeVector} = {@code includeText} = {@code includeMetadata} = true
 * </ul>
 *
 * <p>{@code overQueryFactor} is expressed as a {@code float} so non-integer factors like {@code
 * 1.5f} or {@code 2.5f} are supported. Brute-force backends (e.g., {@link
 * com.integrallis.vectors.db.index.FlatScanAdapter}) ignore both {@code overQueryFactor} and {@code
 * searchListSize}.
 */
public record SearchRequest(
    float[] query,
    int k,
    int searchListSize,
    float overQueryFactor,
    float minScore,
    Filter filter,
    boolean includeVector,
    boolean includeText,
    boolean includeMetadata) {

  public SearchRequest {
    Objects.requireNonNull(query, "query must not be null");
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
  }

  /** Creates a new builder with the given query and k. */
  public static Builder builder(float[] query, int k) {
    return new Builder(query, k);
  }

  /** Mutable builder for {@link SearchRequest}. */
  public static final class Builder {

    private final float[] query;
    private final int k;
    private Integer searchListSize;
    private Float overQueryFactor;
    private Float minScore;
    private Filter filter;
    private boolean includeVector = true;
    private boolean includeText = true;
    private boolean includeMetadata = true;

    private Builder(float[] query, int k) {
      this.query = Objects.requireNonNull(query, "query must not be null");
      if (k <= 0) {
        throw new IllegalArgumentException("k must be positive: " + k);
      }
      this.k = k;
    }

    /** Sets the beam width for the coarse search pass. */
    public Builder searchListSize(int searchListSize) {
      this.searchListSize = searchListSize;
      return this;
    }

    /** Sets the over-query multiplier. Non-integer values such as {@code 1.5f} are allowed. */
    public Builder overQueryFactor(float overQueryFactor) {
      this.overQueryFactor = overQueryFactor;
      return this;
    }

    /** Sets the minimum score threshold. */
    public Builder minScore(float minScore) {
      this.minScore = minScore;
      return this;
    }

    /** Sets the metadata filter (null is allowed and means "match everything"). */
    public Builder filter(Filter filter) {
      this.filter = filter;
      return this;
    }

    /** Include the raw vector in each returned Hit's Document. */
    public Builder includeVector(boolean includeVector) {
      this.includeVector = includeVector;
      return this;
    }

    /** Include the raw text in each returned Hit's Document. */
    public Builder includeText(boolean includeText) {
      this.includeText = includeText;
      return this;
    }

    /** Include the metadata map in each returned Hit's Document. */
    public Builder includeMetadata(boolean includeMetadata) {
      this.includeMetadata = includeMetadata;
      return this;
    }

    /** Constructs the {@link SearchRequest} with defaults applied for any unset fields. */
    public SearchRequest build() {
      int l = searchListSize != null ? searchListSize : Math.max(k, 100);
      float oqf = overQueryFactor != null ? overQueryFactor : 4.0f;
      float ms = minScore != null ? minScore : -Float.MAX_VALUE;
      return new SearchRequest(
          query, k, l, oqf, ms, filter, includeVector, includeText, includeMetadata);
    }
  }
}
