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
package com.integrallis.vectors.studio.core.recommender;

import com.integrallis.vectors.studio.core.connection.CollectionSummary;
import com.integrallis.vectors.studio.core.connection.StudioBackend;
import com.integrallis.vectors.studio.core.search.DocumentView;
import java.util.List;
import smile.feature.extraction.PCA;
import smile.tensor.Vector;

/** Computes {@link DatasetStats} from a {@link StudioBackend}'s view of a collection. */
public final class DatasetStatsCollector {

  private DatasetStatsCollector() {}

  /** Analyses a collection by sampling at most {@code sampleSize} documents. */
  public static DatasetStats analyze(StudioBackend backend, String collection, int sampleSize) {
    CollectionSummary summary = backend.describe(collection);
    List<DocumentView> sample = backend.previewDocuments(collection, 0, sampleSize);
    int dim = summary.dimension();
    double sparsity = sparsityRatio(sample);
    boolean hasText = sample.stream().anyMatch(d -> d.text() != null && !d.text().isBlank());
    double intrinsic = estimateIntrinsicDim(sample, Math.min(dim, 50));
    return new DatasetStats((int) summary.size(), dim, intrinsic, null, hasText, sparsity);
  }

  private static double sparsityRatio(List<DocumentView> sample) {
    long total = 0;
    long zeros = 0;
    for (DocumentView d : sample) {
      if (d.vector() == null) continue;
      total += d.vector().length;
      for (float v : d.vector()) if (v == 0.0f) zeros++;
    }
    return total == 0 ? 0.0 : (double) zeros / total;
  }

  private static double estimateIntrinsicDim(List<DocumentView> sample, int maxComponents) {
    if (sample.size() < 2) return 0.0;
    int n = Math.min(sample.size(), 1024);
    int dim = sample.get(0).vector() == null ? 0 : sample.get(0).vector().length;
    if (dim < 2) return 0.0;
    double[][] data = new double[n][dim];
    for (int i = 0; i < n; i++) {
      DocumentView d = sample.get(i);
      if (d.vector() == null) continue;
      for (int j = 0; j < dim; j++) data[i][j] = d.vector()[j];
    }
    try {
      PCA pca = PCA.fit(data);
      Vector vp = pca.cumulativeVarianceProportion();
      int target = Math.min(vp.size(), maxComponents);
      for (int i = 0; i < target; i++) {
        if (vp.get(i) >= 0.90) return i + 1;
      }
      return target;
    } catch (RuntimeException e) {
      return dim;
    }
  }
}
