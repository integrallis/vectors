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
package com.integrallis.vectors.studio.core.projection;

/** Callback invoked by a {@link Projection} run for incremental progress updates. */
public interface ProgressListener {

  /** Called by the projection at periodic intervals (or once for non-iterative algorithms). */
  void onIteration(int iter, int total, float[][] currentCoords);

  /** Called once when the projection completes successfully. */
  default void onDone(ProjectionResult result) {}

  /** Called once if the projection throws. */
  default void onError(Throwable t) {}

  /** No-op listener. */
  static ProgressListener noop() {
    return (i, t, c) -> {};
  }
}
