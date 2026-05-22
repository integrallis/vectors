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

/**
 * Wire-format SSE event for an optimization study. Mirrors the shape and naming convention of
 * {@code ProjectionEvent}: a sealed interface with a small set of records, each carrying the {@code
 * studyId} so subscribers can multiplex multiple studies on a single channel.
 */
public sealed interface OptimizeEvent {

  String studyId();

  /** Emitted once when the runner picks up the study. */
  record SubmittedEvt(String studyId, int nTrials) implements OptimizeEvent {}

  /** Emitted at the start of each trial. */
  record TrialStartedEvt(String studyId, String trialId, int index) implements OptimizeEvent {}

  /** Emitted when a trial completes successfully. Carries the full {@link TrialResult}. */
  record TrialCompletedEvt(String studyId, TrialResult result, int index)
      implements OptimizeEvent {}

  /** Emitted when a trial fails (the study continues). */
  record TrialFailedEvt(String studyId, String trialId, int index, String message)
      implements OptimizeEvent {}

  /**
   * Periodic progress heartbeat (optional, for clients that want a "trials_completed/total" tick).
   */
  record ProgressEvt(String studyId, int trialsCompleted, int trialsTotal)
      implements OptimizeEvent {}

  /** Emitted once when the trial budget is exhausted. */
  record CompletedEvt(String studyId, int trialsCompleted) implements OptimizeEvent {}

  /** Emitted once when the study was cancelled mid-run. */
  record CancelledEvt(String studyId, int trialsCompleted) implements OptimizeEvent {}

  /** Emitted on a fatal runner error (no further events follow). */
  record ErrorEvt(String studyId, String message) implements OptimizeEvent {}
}
