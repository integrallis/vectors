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

import com.integrallis.vectors.optimizer.persist.StudyStore;
import com.integrallis.vectors.optimizer.sampler.GridSampler;
import com.integrallis.vectors.optimizer.sampler.NoMoreTrialsException;
import com.integrallis.vectors.optimizer.sampler.ParamSampler;
import com.integrallis.vectors.optimizer.sampler.RandomSampler;
import com.integrallis.vectors.optimizer.sampler.TpeSampler;
import com.integrallis.vectors.optimizer.space.Trial;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Flow;
import java.util.concurrent.SubmissionPublisher;

/**
 * Runs a study sequentially: pulls a {@link Trial} from the configured sampler, hands it to
 * {@link IndexStudy}, persists the resulting {@link TrialResult} via {@link StudyStore}, and
 * publishes a {@link TrialEvent} stream so live-progress subscribers (Studio SSE, CLI loggers)
 * see the run unfold.
 *
 * <p>Sequential by design — concurrent trials would compete for memory and JVM warmup, biasing
 * latency measurements. Multi-trial parallelism is a v2 follow-up.
 */
public final class StudyRunner implements AutoCloseable {

  private final String studyId;
  private final IndexStudy study;
  private final StudyConfig cfg;
  private final StudyStore store;
  private final ParamSampler sampler;
  private final SubmissionPublisher<TrialEvent> publisher;
  private volatile boolean cancelled;

  public StudyRunner(String studyId, IndexStudy study, StudyConfig cfg, StudyStore store) {
    this.studyId = Objects.requireNonNull(studyId, "studyId");
    this.study = Objects.requireNonNull(study, "study");
    this.cfg = Objects.requireNonNull(cfg, "cfg");
    this.store = Objects.requireNonNull(store, "store");
    this.sampler = newSampler(cfg);
    this.publisher = new SubmissionPublisher<>();
  }

  /** Subscribe-able stream of {@link TrialEvent}s for live progress reporting. */
  public Flow.Publisher<TrialEvent> publisher() {
    return publisher;
  }

  /** Caller-controlled cooperative cancel. The runner exits between trials. */
  public void cancel() {
    cancelled = true;
  }

  /** Synchronously executes the trial budget. Safe to call from a virtual thread. */
  public void runBlocking() {
    publisher.submit(new TrialEvent.Started(studyId, cfg.nTrials(), Instant.now()));
    int completed = 0;
    try {
      List<TrialResult> history = new ArrayList<>();
      for (int i = 0; i < cfg.nTrials(); i++) {
        if (cancelled) {
          publisher.submit(new TrialEvent.Cancelled(studyId, completed, Instant.now()));
          return;
        }
        Trial trial;
        try {
          trial = sampler.next(history);
        } catch (NoMoreTrialsException ex) {
          // Grid exhausted before budget — terminate cleanly.
          break;
        }
        publisher.submit(new TrialEvent.TrialStarted(studyId, trial, i, Instant.now()));
        try {
          TrialResult tr = study.runOne(trial);
          history.add(tr);
          store.appendTrial(studyId, tr);
          publisher.submit(new TrialEvent.TrialCompleted(studyId, tr, i));
          completed++;
        } catch (RuntimeException ex) {
          publisher.submit(new TrialEvent.TrialFailed(studyId, trial, i, ex.toString()));
        }
      }
      publisher.submit(new TrialEvent.Completed(studyId, completed, Instant.now()));
    } catch (RuntimeException fatal) {
      publisher.submit(new TrialEvent.Error(studyId, fatal.toString(), Instant.now()));
      throw fatal;
    }
  }

  @Override
  public void close() {
    publisher.close();
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
}
