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
package com.integrallis.vectors.studio.web.optimize;

import com.integrallis.vectors.optimizer.embed.EmbeddingProvider;
import com.integrallis.vectors.optimizer.persist.StudyStore;
import com.integrallis.vectors.optimizer.sampler.GridSampler;
import com.integrallis.vectors.optimizer.sampler.NoMoreTrialsException;
import com.integrallis.vectors.optimizer.sampler.ParamSampler;
import com.integrallis.vectors.optimizer.sampler.RandomSampler;
import com.integrallis.vectors.optimizer.sampler.TpeSampler;
import com.integrallis.vectors.optimizer.space.Trial;
import com.integrallis.vectors.optimizer.study.IndexStudy;
import com.integrallis.vectors.optimizer.study.StudyConfig;
import com.integrallis.vectors.optimizer.study.TrialResult;
import java.time.Duration;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Lifecycle manager for {@link OptimizeJob}s. Submits each study to a virtual-thread executor,
 * tracks them in a {@link ConcurrentHashMap}, evicts non-running studies after an idle TTL, and
 * persists every trial via the optimizer's {@link StudyStore}. Modeled exactly on {@code
 * ProjectionJobManager}.
 */
public final class OptimizeJobManager implements AutoCloseable {

  private static final Duration DEFAULT_IDLE_TTL = Duration.ofMinutes(10);

  private final ConcurrentHashMap<String, OptimizeJob> jobs = new ConcurrentHashMap<>();
  private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
  private final ScheduledExecutorService janitor =
      Executors.newSingleThreadScheduledExecutor(r -> Thread.ofVirtual().unstarted(r));
  private final StudyStore store;
  private final Duration idleTtl;

  public OptimizeJobManager() {
    this(StudyStore.defaultRoot(), DEFAULT_IDLE_TTL);
  }

  public OptimizeJobManager(StudyStore store, Duration idleTtl) {
    this.store = store;
    this.idleTtl = idleTtl;
    janitor.scheduleAtFixedRate(this::expireIdle, 1, 1, TimeUnit.MINUTES);
  }

  /** Submit an Index study. Returns the assigned study id; the run executes on a virtual thread. */
  public String submit(StudyConfig cfg, EmbeddingProvider embedder) {
    String studyId = UUID.randomUUID().toString();
    OptimizeJob job = new OptimizeJob(studyId, cfg.nTrials());
    jobs.put(studyId, job);
    IndexStudy study = new IndexStudy(cfg, embedder);
    executor.submit(() -> runStudy(job, cfg, study));
    return studyId;
  }

  public OptimizeJob get(String studyId) {
    OptimizeJob job = jobs.get(studyId);
    if (job != null) job.touch();
    return job;
  }

  public boolean cancel(String studyId) {
    OptimizeJob j = jobs.get(studyId);
    return j != null && j.cancel();
  }

  @Override
  public void close() {
    janitor.shutdownNow();
    jobs.values().forEach(OptimizeJob::cancel);
    executor.shutdownNow();
    jobs.clear();
  }

  // --- internals ---

  private void runStudy(OptimizeJob job, StudyConfig cfg, IndexStudy study) {
    try {
      job.markRunning();
      ParamSampler sampler = newSampler(cfg);
      for (int i = 0; i < cfg.nTrials(); i++) {
        if (job.isCancelled()) {
          job.markCancelled();
          return;
        }
        Trial trial;
        try {
          trial = sampler.next(job.history());
        } catch (NoMoreTrialsException ex) {
          break;
        }
        try {
          TrialResult tr = study.runOne(trial);
          store.appendTrial(job.studyId(), tr);
          job.recordTrial(tr, i);
        } catch (RuntimeException ex) {
          job.recordTrialFailure(trial.trialId(), i, ex.toString());
        }
      }
      if (!job.isCancelled()) job.markCompleted();
      else job.markCancelled();
    } catch (Throwable t) {
      job.markFailed(t);
    }
  }

  private static ParamSampler newSampler(StudyConfig cfg) {
    return switch (cfg.samplerKind()) {
      case GRID -> new GridSampler(cfg.searchSpace());
      case RANDOM -> new RandomSampler(cfg.searchSpace(), cfg.seed());
      case TPE -> {
        TpeSampler.Hyperparameters hp =
            cfg.tpeHp() != null ? cfg.tpeHp() : TpeSampler.Hyperparameters.defaults();
        yield new TpeSampler(cfg.searchSpace(), cfg.seed(), hp);
      }
    };
  }

  // Package-private so tests can trigger eviction deterministically without
  // waiting on the 1-minute janitor cadence.
  void expireIdle() {
    long cutoff = System.currentTimeMillis() - idleTtl.toMillis();
    jobs.values().removeIf(j -> j.lastTouched() < cutoff && j.state() != OptimizeJob.State.RUNNING);
  }
}
