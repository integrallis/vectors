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

import com.integrallis.vectors.studio.core.projection.ProjectionResult;

/** Sealed event hierarchy emitted by a {@link ProjectionJob} during its lifecycle. */
public sealed interface ProjectionEvent {

  /** Periodic progress update (per-iteration coordinates). */
  record Progress(String jobId, int iter, int total, float[][] coords) implements ProjectionEvent {}

  /** Final completion event carrying the full result. */
  record Done(String jobId, ProjectionResult result) implements ProjectionEvent {}

  /** Terminal error. */
  record Error(String jobId, String message) implements ProjectionEvent {}
}
