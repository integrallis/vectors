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
package com.integrallis.vectors.db;

import com.integrallis.vectors.core.filter.Filter;
import java.util.Objects;

/**
 * A search request. Immutable; use the {@link Builder} for fluent construction with defaults.
 *
 * <p>Defaults applied by {@link Builder#build()}:
 *
 * <ul>
 *   <li>{@code searchListSize} = {@code max(k, 100)}
 *   <li>{@code overQueryFactor} = 4.0f
 *   <li>{@code filterExpansion} = 4.0f
 *   <li>{@code minScore} = {@code -Float.MAX_VALUE} (no minimum)
 *   <li>{@code filter} = null (no filter; semantically equivalent to {@link
 *       com.integrallis.vectors.core.filter.Filters#all()})
 *   <li>{@code includeVector} = {@code includeText} = {@code includeMetadata} = true
 *   <li>{@code searchMultiStart} = 1 (single-start beam search)
 * </ul>
 *
 * <p><b>Two independent multipliers.</b> {@code overQueryFactor} controls how many extra candidates
 * the SPI's two-pass quantized search path retrieves for rescoring (it is passed through to the
 * graph index; quantized flat scan also uses it to size the compressed preselection before exact
 * rescoring). {@code filterExpansion} controls how many extra candidates the facade requests from
 * the SPI when a metadata filter is active, so that post-filtering still yields {@code k} results
 * in most cases. The two do not compound: the SPI always receives the original {@code k}, and the
 * facade requests {@code k * filterExpansion} candidates from the SPI only when a non-trivial
 * filter is present.
 *
 * <p>{@code overQueryFactor} and {@code filterExpansion} are both expressed as {@code float} so
 * non-integer factors like {@code 1.5f} or {@code 2.5f} are supported. Unquantized brute-force
 * backends (e.g., {@link com.integrallis.vectors.db.index.FlatScanAdapter}) ignore both {@code
 * overQueryFactor} and {@code searchListSize}.
 */
public record SearchRequest(
    float[] query,
    int k,
    int searchListSize,
    float overQueryFactor,
    float filterExpansion,
    float minScore,
    Filter filter,
    boolean includeVector,
    boolean includeText,
    boolean includeMetadata,
    int searchMultiStart) {

  public SearchRequest {
    Objects.requireNonNull(query, "query must not be null");
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
    if (searchMultiStart < 1) {
      throw new IllegalArgumentException("searchMultiStart must be >= 1: " + searchMultiStart);
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
    private Float filterExpansion;
    private Float minScore;
    private Filter filter;
    private boolean includeVector = true;
    private boolean includeText = true;
    private boolean includeMetadata = true;
    private Integer searchMultiStart;

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

    /**
     * Sets the over-query multiplier for the SPI's two-pass quantized search path. This controls
     * how many coarse-pass candidates are rescored at full precision. Has no effect on the
     * post-filter candidate pool; use {@link #filterExpansion(float)} for that. Non-integer values
     * such as {@code 1.5f} are allowed.
     */
    public Builder overQueryFactor(float overQueryFactor) {
      this.overQueryFactor = overQueryFactor;
      return this;
    }

    /**
     * Sets the candidate pool multiplier for post-filter expansion. When a non-trivial metadata
     * filter is active, the facade requests {@code k * filterExpansion} candidates from the SPI so
     * that post-filtering still yields {@code k} results. Default: {@code 4.0f}. Independent of
     * {@link #overQueryFactor(float)}, which controls quantization rescoring.
     */
    public Builder filterExpansion(float filterExpansion) {
      this.filterExpansion = filterExpansion;
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

    /**
     * Include the metadata map in each returned Hit's Document. Default: {@code true}.
     *
     * <p>When {@code false}, the metadata map in the returned documents will be {@code null}. Note
     * that metadata filters (via {@link #filter(Filter)}) still evaluate correctly against the
     * stored metadata regardless of this setting — this flag only affects what is projected into
     * the response.
     */
    public Builder includeMetadata(boolean includeMetadata) {
      this.includeMetadata = includeMetadata;
      return this;
    }

    /**
     * Sets the number of parallel seed starts for multi-start beam search on graph-based backends.
     * Default {@code 1} runs a single-start beam search (bit-identical to the legacy path). Values
     * {@code > 1} dispatch {@code searchMultiStart} independent beam searches on virtual threads
     * and merge their top-k outputs; composes with {@link VectorCollection#searchBatch} for a batch
     * × multi-start (N×M) parallelism product.
     *
     * <p>Non-graph backends (flat scan, IVF) ignore this parameter. The ACORN pre-filter path also
     * ignores this parameter.
     *
     * @throws IllegalArgumentException if {@code searchMultiStart < 1}
     */
    public Builder searchMultiStart(int searchMultiStart) {
      if (searchMultiStart < 1) {
        throw new IllegalArgumentException("searchMultiStart must be >= 1: " + searchMultiStart);
      }
      this.searchMultiStart = searchMultiStart;
      return this;
    }

    /** Constructs the {@link SearchRequest} with defaults applied for any unset fields. */
    public SearchRequest build() {
      int l = searchListSize != null ? searchListSize : Math.max(k, 100);
      float oqf = overQueryFactor != null ? overQueryFactor : 4.0f;
      float fe = filterExpansion != null ? filterExpansion : 4.0f;
      float ms = minScore != null ? minScore : -Float.MAX_VALUE;
      int sms = searchMultiStart != null ? searchMultiStart : 1;
      return new SearchRequest(
          query, k, l, oqf, fe, ms, filter, includeVector, includeText, includeMetadata, sms);
    }
  }
}
