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

import com.integrallis.vectors.optimizer.study.TrialResult;
import java.time.Instant;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.SubmissionPublisher;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Per-study runtime handle: state machine, accumulated trial history, event publisher, and idle
 * timestamp used by {@link OptimizeJobManager} for TTL-based eviction. Mirrors {@code
 * ProjectionJob}.
 */
public final class OptimizeJob {

  /** Lifecycle state. */
  public enum State {
    PENDING,
    RUNNING,
    COMPLETED,
    CANCELLED,
    FAILED
  }

  private final String studyId;
  private final Instant submittedAt;
  private final int nTrials;
  private final SubmissionPublisher<OptimizeEvent> publisher = new SubmissionPublisher<>();
  private final AtomicReference<State> state = new AtomicReference<>(State.PENDING);
  private final AtomicReference<Throwable> error = new AtomicReference<>();
  private final AtomicLong lastTouched = new AtomicLong(System.currentTimeMillis());
  private final List<TrialResult> history = new CopyOnWriteArrayList<>();
  private volatile boolean cancelled;

  public OptimizeJob(String studyId, int nTrials) {
    this.studyId = studyId;
    this.nTrials = nTrials;
    this.submittedAt = Instant.now();
  }

  public String studyId() {
    return studyId;
  }

  public Instant submittedAt() {
    return submittedAt;
  }

  public int nTrials() {
    return nTrials;
  }

  public State state() {
    return state.get();
  }

  public Throwable error() {
    return error.get();
  }

  public long lastTouched() {
    return lastTouched.get();
  }

  /** Read-only snapshot of completed trials. */
  public List<TrialResult> history() {
    return List.copyOf(history);
  }

  public SubmissionPublisher<OptimizeEvent> publisher() {
    return publisher;
  }

  /** Caller-controlled cooperative cancel. */
  public boolean cancel() {
    cancelled = true;
    return state.compareAndSet(State.PENDING, State.CANCELLED)
        || state.compareAndSet(State.RUNNING, State.CANCELLED);
  }

  public boolean isCancelled() {
    return cancelled;
  }

  // --- transitions used by OptimizeJobManager only ---

  void markRunning() {
    state.compareAndSet(State.PENDING, State.RUNNING);
    publisher.submit(new OptimizeEvent.SubmittedEvt(studyId, nTrials));
    touch();
  }

  void recordTrial(TrialResult tr, int index) {
    history.add(tr);
    publisher.submit(new OptimizeEvent.TrialCompletedEvt(studyId, tr, index));
    publisher.submit(new OptimizeEvent.ProgressEvt(studyId, history.size(), nTrials));
    touch();
  }

  void recordTrialFailure(String trialId, int index, String message) {
    publisher.submit(new OptimizeEvent.TrialFailedEvt(studyId, trialId, index, message));
    touch();
  }

  void markCompleted() {
    state.compareAndSet(State.RUNNING, State.COMPLETED);
    publisher.submit(new OptimizeEvent.CompletedEvt(studyId, history.size()));
    publisher.close();
    touch();
  }

  void markCancelled() {
    publisher.submit(new OptimizeEvent.CancelledEvt(studyId, history.size()));
    publisher.close();
    touch();
  }

  void markFailed(Throwable t) {
    error.set(t);
    state.compareAndSet(State.RUNNING, State.FAILED);
    state.compareAndSet(State.PENDING, State.FAILED);
    publisher.submit(new OptimizeEvent.ErrorEvt(studyId, t.toString()));
    publisher.close();
    touch();
  }

  void touch() {
    lastTouched.set(System.currentTimeMillis());
  }
}
