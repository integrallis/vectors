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
package com.integrallis.vectors.studio.web.dataset;

import java.util.concurrent.SubmissionPublisher;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * One outstanding dataset load. Subscribers consume {@link DatasetLoadEvent}s through {@link
 * #publisher()}. State transitions: RUNNING &rarr; DONE | ERROR.
 */
public final class DatasetLoadJob {

  /** Lifecycle state. */
  public enum State {
    RUNNING,
    DONE,
    ERROR
  }

  private final String jobId;
  private final String collectionName;
  private final int total;
  private final SubmissionPublisher<DatasetLoadEvent> publisher = new SubmissionPublisher<>();
  private final AtomicReference<State> state = new AtomicReference<>(State.RUNNING);
  private final AtomicInteger loaded = new AtomicInteger();
  private final AtomicReference<String> error = new AtomicReference<>();
  private final AtomicLong lastTouched = new AtomicLong(System.currentTimeMillis());

  public DatasetLoadJob(String jobId, String collectionName, int total) {
    this.jobId = jobId;
    this.collectionName = collectionName;
    this.total = total;
  }

  public String jobId() {
    return jobId;
  }

  public String collectionName() {
    return collectionName;
  }

  public int total() {
    return total;
  }

  public int loaded() {
    return loaded.get();
  }

  public State state() {
    return state.get();
  }

  public String error() {
    return error.get();
  }

  public long lastTouched() {
    return lastTouched.get();
  }

  public SubmissionPublisher<DatasetLoadEvent> publisher() {
    return publisher;
  }

  /** Snapshot of the current state as an SSE event (no terminal side effects). */
  public DatasetLoadEvent snapshot() {
    return new DatasetLoadEvent(jobId, loaded.get(), total, state.get().name(), error.get());
  }

  /** Records the running loaded count and publishes a RUNNING progress event. */
  public void progress(int loadedCount) {
    if (state.get() != State.RUNNING) {
      return;
    }
    loaded.set(loadedCount);
    lastTouched.set(System.currentTimeMillis());
    publisher.submit(new DatasetLoadEvent(jobId, loadedCount, total, State.RUNNING.name(), null));
  }

  /**
   * Marks DONE, publishes the terminal event with the final loaded count, and closes the stream.
   */
  public void complete(int finalCount) {
    loaded.set(finalCount);
    lastTouched.set(System.currentTimeMillis());
    if (state.compareAndSet(State.RUNNING, State.DONE)) {
      publisher.submit(new DatasetLoadEvent(jobId, finalCount, total, State.DONE.name(), null));
      publisher.close();
    }
  }

  /** Marks ERROR, publishes the terminal event, and closes the stream. */
  public void fail(String message) {
    error.set(message);
    lastTouched.set(System.currentTimeMillis());
    if (state.compareAndSet(State.RUNNING, State.ERROR)) {
      publisher.submit(
          new DatasetLoadEvent(jobId, loaded.get(), total, State.ERROR.name(), message));
      publisher.close();
    }
  }
}
