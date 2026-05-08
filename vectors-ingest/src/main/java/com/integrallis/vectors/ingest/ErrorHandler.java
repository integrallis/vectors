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
package com.integrallis.vectors.ingest;

/**
 * Strategy for handling permanent (non-retryable, or retry-exhausted) failures during ingestion.
 * The default is {@link #failFast()} — abort the run and surface the throwable to the caller. A
 * {@link #continueOnError(java.util.function.Consumer)} variant lets callers log-and-skip while the
 * rest of the source still drains.
 */
@FunctionalInterface
public interface ErrorHandler {

  /** Decision returned by {@link #onError}. */
  enum Decision {
    FAIL_FAST,
    CONTINUE
  }

  /** Called for every permanent error. Return {@link Decision#FAIL_FAST} to abort. */
  Decision onError(IngestErrorContext ctx);

  /** Default fail-fast handler. */
  static ErrorHandler failFast() {
    return ctx -> Decision.FAIL_FAST;
  }

  /** Continue-on-error handler that hands every failure to {@code observer}. */
  static ErrorHandler continueOnError(java.util.function.Consumer<IngestErrorContext> observer) {
    return ctx -> {
      observer.accept(ctx);
      return Decision.CONTINUE;
    };
  }

  /** Bag of context attached to a failure. */
  record IngestErrorContext(String stage, long batchId, Throwable error) {}
}
