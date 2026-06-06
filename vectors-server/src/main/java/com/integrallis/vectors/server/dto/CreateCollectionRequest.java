/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
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
 * <p>Only {@code name}, {@code dimension}, {@code metric}, and {@code indexType} are required;
 * everything else is an optional tuning knob that defaults to the {@link VectorCollectionBuilder}
 * default when unset.
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
 *   "hnswBuildThreads":     8,
 *   "vamanaMaxDegree":      64,
 *   "vamanaSearchListSize": 128,
 *   "vamanaAlpha":          1.2,
 *   "vamanaBuildThreads":   8,
 *   "ivfK":                 1024,
 *   "ivfNprobe":            32,
 *   "ivfMaxIter":           30,
 *   "ivfPqSubspaces":       16,
 *   "ivfPqClusters":        256,
 *   "ivfRescoreFactor":     4,
 *   "autoCommitThreshold":  32
 * }
 * }</pre>
 *
 * <p>The index-specific knobs are applied unconditionally; {@link VectorCollectionBuilder} ignores
 * the ones that don't apply to the chosen {@code indexType} (e.g. {@code vamanaMaxDegree} is a
 * no-op for HNSW). Enum strings are matched case-insensitively. Unknown enum values or out-of-range
 * numbers produce a {@code 400 Bad Request}.
 */
public record CreateCollectionRequest(
    String name,
    Integer dimension,
    String metric,
    String indexType,
    String quantizer,
    Integer hnswM,
    Integer hnswEfConstruction,
    Integer hnswBuildThreads,
    Integer vamanaMaxDegree,
    Integer vamanaSearchListSize,
    Double vamanaAlpha,
    Integer vamanaBuildThreads,
    Integer ivfK,
    Integer ivfNprobe,
    Integer ivfMaxIter,
    Integer ivfPqSubspaces,
    Integer ivfPqClusters,
    Integer ivfRescoreFactor,
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
      @JsonProperty("hnswBuildThreads") Integer hnswBuildThreads,
      @JsonProperty("vamanaMaxDegree") Integer vamanaMaxDegree,
      @JsonProperty("vamanaSearchListSize") Integer vamanaSearchListSize,
      @JsonProperty("vamanaAlpha") Double vamanaAlpha,
      @JsonProperty("vamanaBuildThreads") Integer vamanaBuildThreads,
      @JsonProperty("ivfK") Integer ivfK,
      @JsonProperty("ivfNprobe") Integer ivfNprobe,
      @JsonProperty("ivfMaxIter") Integer ivfMaxIter,
      @JsonProperty("ivfPqSubspaces") Integer ivfPqSubspaces,
      @JsonProperty("ivfPqClusters") Integer ivfPqClusters,
      @JsonProperty("ivfRescoreFactor") Integer ivfRescoreFactor,
      @JsonProperty("autoCommitThreshold") Integer autoCommitThreshold) {
    this.name = name;
    this.dimension = dimension;
    this.metric = metric;
    this.indexType = indexType;
    this.quantizer = quantizer;
    this.hnswM = hnswM;
    this.hnswEfConstruction = hnswEfConstruction;
    this.hnswBuildThreads = hnswBuildThreads;
    this.vamanaMaxDegree = vamanaMaxDegree;
    this.vamanaSearchListSize = vamanaSearchListSize;
    this.vamanaAlpha = vamanaAlpha;
    this.vamanaBuildThreads = vamanaBuildThreads;
    this.ivfK = ivfK;
    this.ivfNprobe = ivfNprobe;
    this.ivfMaxIter = ivfMaxIter;
    this.ivfPqSubspaces = ivfPqSubspaces;
    this.ivfPqClusters = ivfPqClusters;
    this.ivfRescoreFactor = ivfRescoreFactor;
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
    String positive = firstNonPositive();
    if (positive != null) {
      return positive;
    }
    if (vamanaAlpha != null && vamanaAlpha < 1.0) {
      return "vamanaAlpha must be >= 1.0";
    }
    return null;
  }

  /** Returns an error message for the first int knob that is set but non-positive, else null. */
  private String firstNonPositive() {
    String[] names = {
      "hnswM", "hnswEfConstruction", "hnswBuildThreads",
      "vamanaMaxDegree", "vamanaSearchListSize", "vamanaBuildThreads",
      "ivfK", "ivfNprobe", "ivfMaxIter",
      "ivfPqSubspaces", "ivfPqClusters", "ivfRescoreFactor",
      "autoCommitThreshold"
    };
    Integer[] values = {
      hnswM, hnswEfConstruction, hnswBuildThreads,
      vamanaMaxDegree, vamanaSearchListSize, vamanaBuildThreads,
      ivfK, ivfNprobe, ivfMaxIter,
      ivfPqSubspaces, ivfPqClusters, ivfRescoreFactor,
      autoCommitThreshold
    };
    for (int i = 0; i < names.length; i++) {
      if (values[i] != null && values[i] <= 0) {
        return names[i] + " must be positive";
      }
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
    // HNSW knobs.
    if (hnswM != null) {
      builder.hnswM(hnswM);
    }
    if (hnswEfConstruction != null) {
      builder.hnswEfConstruction(hnswEfConstruction);
    }
    if (hnswBuildThreads != null) {
      builder.hnswBuildThreads(hnswBuildThreads);
    }
    // Vamana knobs (search-time L is supplied per query via efSearch).
    if (vamanaMaxDegree != null) {
      builder.vamanaMaxDegree(vamanaMaxDegree);
    }
    if (vamanaSearchListSize != null) {
      builder.vamanaSearchListSize(vamanaSearchListSize);
    }
    if (vamanaAlpha != null) {
      builder.vamanaAlpha(vamanaAlpha.floatValue());
    }
    if (vamanaBuildThreads != null) {
      builder.vamanaBuildThreads(vamanaBuildThreads);
    }
    // IVF / IVF-PQ knobs (nprobe is a build-time setting in vectors, swept by rebuilding).
    if (ivfK != null) {
      builder.ivfK(ivfK);
    }
    if (ivfNprobe != null) {
      builder.ivfNprobe(ivfNprobe);
    }
    if (ivfMaxIter != null) {
      builder.ivfMaxIter(ivfMaxIter);
    }
    if (ivfPqSubspaces != null) {
      builder.ivfPqSubspaces(ivfPqSubspaces);
    }
    if (ivfPqClusters != null) {
      builder.ivfPqClusters(ivfPqClusters);
    }
    if (ivfRescoreFactor != null) {
      builder.ivfRescoreFactor(ivfRescoreFactor);
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
