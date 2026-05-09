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
package com.integrallis.vectors.optimizer.study;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.optimizer.data.Qrels;
import com.integrallis.vectors.optimizer.data.Queries;
import com.integrallis.vectors.optimizer.objective.ObjectiveWeights;
import com.integrallis.vectors.optimizer.sampler.TpeSampler;
import com.integrallis.vectors.optimizer.space.SearchSpace;
import java.util.List;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Top-level study configuration. Carries both the JSON-friendly knobs (search space, weights,
 * sampler choice, trial budget) and the lazy data sources (corpus, qrels, queries) required to
 * execute a study. The data sources are intentionally suppliers so the caller can defer expensive
 * loads (e.g. cloning a {@code VectorCollection}'s documents) until the runner actually starts.
 */
public record StudyConfig(
    SearchSpace searchSpace,
    ObjectiveWeights objectiveWeights,
    SamplerKind samplerKind,
    TpeSampler.Hyperparameters tpeHp,
    Supplier<List<Document>> corpusSource,
    Supplier<Qrels> qrelsSource,
    Supplier<Queries> queriesSource,
    int nTrials,
    int kForMetrics,
    long seed,
    int warmupRounds,
    int measurementRounds,
    boolean gcBetweenTrials) {

  /** Selects which {@link com.integrallis.vectors.optimizer.sampler.ParamSampler} to use. */
  public enum SamplerKind {
    GRID,
    RANDOM,
    TPE
  }

  public StudyConfig {
    Objects.requireNonNull(searchSpace, "searchSpace");
    Objects.requireNonNull(objectiveWeights, "objectiveWeights");
    Objects.requireNonNull(samplerKind, "samplerKind");
    Objects.requireNonNull(corpusSource, "corpusSource");
    Objects.requireNonNull(qrelsSource, "qrelsSource");
    Objects.requireNonNull(queriesSource, "queriesSource");
    if (nTrials < 1) throw new IllegalArgumentException("nTrials >= 1");
    if (kForMetrics < 1) throw new IllegalArgumentException("kForMetrics >= 1");
    if (warmupRounds < 0) throw new IllegalArgumentException("warmupRounds >= 0");
    if (measurementRounds < 1) throw new IllegalArgumentException("measurementRounds >= 1");
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Mutable builder with the same defaults documented in the plan. */
  public static final class Builder {
    private SearchSpace searchSpace;
    private ObjectiveWeights objectiveWeights;
    private SamplerKind samplerKind = SamplerKind.RANDOM;
    private TpeSampler.Hyperparameters tpeHp;
    private Supplier<List<Document>> corpusSource;
    private Supplier<Qrels> qrelsSource;
    private Supplier<Queries> queriesSource;
    private int nTrials = 20;
    private int kForMetrics = 10;
    private long seed = 0L;
    private int warmupRounds = 3;
    private int measurementRounds = 5;
    private boolean gcBetweenTrials = false;

    public Builder searchSpace(SearchSpace s) { this.searchSpace = s; return this; }
    public Builder objectiveWeights(ObjectiveWeights w) { this.objectiveWeights = w; return this; }
    public Builder samplerKind(SamplerKind s) { this.samplerKind = s; return this; }
    public Builder tpeHp(TpeSampler.Hyperparameters hp) { this.tpeHp = hp; return this; }
    public Builder corpusSource(Supplier<List<Document>> s) { this.corpusSource = s; return this; }
    public Builder qrelsSource(Supplier<Qrels> s) { this.qrelsSource = s; return this; }
    public Builder queriesSource(Supplier<Queries> s) { this.queriesSource = s; return this; }
    public Builder nTrials(int n) { this.nTrials = n; return this; }
    public Builder kForMetrics(int k) { this.kForMetrics = k; return this; }
    public Builder seed(long s) { this.seed = s; return this; }
    public Builder warmupRounds(int r) { this.warmupRounds = r; return this; }
    public Builder measurementRounds(int r) { this.measurementRounds = r; return this; }
    public Builder gcBetweenTrials(boolean b) { this.gcBetweenTrials = b; return this; }

    public StudyConfig build() {
      return new StudyConfig(
          searchSpace,
          objectiveWeights,
          samplerKind,
          tpeHp,
          corpusSource,
          qrelsSource,
          queriesSource,
          nTrials,
          kForMetrics,
          seed,
          warmupRounds,
          measurementRounds,
          gcBetweenTrials);
    }
  }
}
