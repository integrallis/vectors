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
package com.integrallis.vectors.ivf;

/**
 * How {@link BuoyIndex} trains its centroids.
 *
 * <p>{@link #K_MEANS} is the default: Lloyd's algorithm for {@link IvfBuildParams#maxIter()}
 * iterations, the trainer every IVF index was built with through 0.1.34 and the one the committed
 * recall baseline was recorded against. {@link #E_MEANS} is opt-in: the mini-batch EMA trainer
 * ported from FAISS {@code contrib/e_means} (Bessa et al., 2026) at its five-epoch default. It
 * trains in 0.19–0.71× the time of 50-iteration k-means, but on the audit corpus its partitions
 * reached lower recall at a fixed {@code nprobe} and were 2–16% slower at matched recall ({@code
 * benchmark-results/2026-11-14-0.1.35-release-audit}); its scaling laws were fitted for n &ge;
 * 10<sup>5</sup>, k &ge; 256 and high intrinsic dimension, below which its own authors call it "not
 * optimal". Choose it deliberately for large, high-dimensional, build-time-bound corpora.
 */
public enum CentroidTrainer {
  /** Lloyd's k-means for {@code maxIter} iterations. The default. */
  K_MEANS,
  /** FAISS e-means, five epochs. Opt-in. */
  E_MEANS
}
