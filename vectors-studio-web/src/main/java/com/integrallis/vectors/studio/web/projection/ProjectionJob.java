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
package com.integrallis.vectors.studio.web.projection;

import com.integrallis.vectors.studio.core.projection.ProgressListener;
import com.integrallis.vectors.studio.core.projection.ProjectionResult;
import java.util.concurrent.SubmissionPublisher;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * One outstanding projection. Subscribers consume {@link ProjectionEvent}s through {@link
 * #publisher()}. State transitions: RUNNING → DONE | ERROR | CANCELLED.
 */
public final class ProjectionJob {

  /** Lifecycle state. */
  public enum State {
    RUNNING,
    DONE,
    ERROR,
    CANCELLED
  }

  private final String jobId;
  private final String[] ids;
  private final SubmissionPublisher<ProjectionEvent> publisher = new SubmissionPublisher<>();
  private final AtomicReference<State> state = new AtomicReference<>(State.RUNNING);
  private final AtomicReference<ProjectionResult> result = new AtomicReference<>();
  private final AtomicLong lastTouched = new AtomicLong(System.currentTimeMillis());

  public ProjectionJob(String jobId, String[] ids) {
    this.jobId = jobId;
    this.ids = ids;
  }

  public String jobId() {
    return jobId;
  }

  public String[] ids() {
    return ids;
  }

  public State state() {
    return state.get();
  }

  public ProjectionResult result() {
    return result.get();
  }

  public long lastTouched() {
    return lastTouched.get();
  }

  public SubmissionPublisher<ProjectionEvent> publisher() {
    return publisher;
  }

  /** Returns a {@link ProgressListener} that pushes events into {@link #publisher()}. */
  public ProgressListener listener() {
    return new ProgressListener() {
      @Override
      public void onIteration(int iter, int total, float[][] coords) {
        if (state.get() != State.CANCELLED) {
          publisher.submit(new ProjectionEvent.Progress(jobId, iter, total, coords));
          lastTouched.set(System.currentTimeMillis());
        }
      }
    };
  }

  /** Marks DONE and publishes the {@link ProjectionEvent.Done} terminal event. */
  public void complete(ProjectionResult r) {
    result.set(r);
    if (state.compareAndSet(State.RUNNING, State.DONE)) {
      publisher.submit(new ProjectionEvent.Done(jobId, r));
      publisher.close();
    }
  }

  /** Marks ERROR and publishes the {@link ProjectionEvent.Error} terminal event. */
  public void fail(Throwable t) {
    if (state.compareAndSet(State.RUNNING, State.ERROR)) {
      publisher.submit(new ProjectionEvent.Error(jobId, t.getMessage()));
      publisher.close();
    }
  }

  /** Marks CANCELLED. Returns true on first cancel, false if already terminal. */
  public boolean cancel() {
    if (state.compareAndSet(State.RUNNING, State.CANCELLED)) {
      publisher.close();
      return true;
    }
    return false;
  }
}
