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

import com.integrallis.vectors.optimizer.space.Trial;
import java.time.Instant;

/**
 * Stream-of-events emitted by {@link StudyRunner} for live progress reporting. Mirrors the shape of
 * {@code ProjectionEvent} in {@code vectors-studio-web}: a sealed interface with a small set of
 * records, each carrying the {@code studyId} so subscribers can multiplex multiple studies on a
 * single channel.
 */
public sealed interface TrialEvent {

  String studyId();

  /** Emitted once when the runner starts. */
  record Started(String studyId, int nTrials, Instant at) implements TrialEvent {}

  /** Emitted when a trial begins execution. */
  record TrialStarted(String studyId, Trial trial, int index, Instant at) implements TrialEvent {}

  /** Emitted on successful trial completion. */
  record TrialCompleted(String studyId, TrialResult result, int index) implements TrialEvent {}

  /** Emitted when a trial fails (the runner continues with the next trial). */
  record TrialFailed(String studyId, Trial trial, int index, String message)
      implements TrialEvent {}

  /** Emitted once when the runner finishes the trial budget. */
  record Completed(String studyId, int trialsCompleted, Instant at) implements TrialEvent {}

  /** Emitted once when {@link StudyRunner#cancel()} interrupts the run. */
  record Cancelled(String studyId, int trialsCompleted, Instant at) implements TrialEvent {}

  /** Emitted on a fatal runner error (no further events follow). */
  record Error(String studyId, String message, Instant at) implements TrialEvent {}
}
