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
package com.integrallis.vectors.core;

final class ClosedNestedDispatchProbe {
  public static void main(String[] args) {
    var closed = new GgufPersistentRowExecutor(2, 1, "closed");
    closed.close();
    try (var active = new GgufPersistentRowExecutor(2, 1, "active")) {
      active.forEach(
          2,
          row -> {
            try {
              closeFromCallback(active);
              throw new AssertionError("callback closed its active executor");
            } catch (IllegalStateException expected) {
            }
            try {
              closed.forEach(1, ignored -> {});
              throw new AssertionError("closed nested row executor accepted work");
            } catch (IllegalStateException expected) {
            }
            try {
              closed.execute(GgufStagePlan.of(GgufStagePlan.stage(1, (a, b) -> {})));
              throw new AssertionError("closed nested stage executor accepted work");
            } catch (IllegalStateException expected) {
            }
          });
    }
  }

  private static void closeFromCallback(GgufRowExecutor executor) {
    executor.close();
  }
}
