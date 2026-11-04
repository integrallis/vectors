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

/**
 * Whether the current thread is inside a GGUF row callback, for every executor in this JVM.
 *
 * <p>One shared answer, not one per executor. A callback that reaches a <i>second</i> executor must
 * be seen by that executor as nested too: if it were not, the second pool would publish work to its
 * own workers, and those workers can re-enter the first pool and leave the two waiting on each
 * other. The flag therefore has to be readable across implementations, which is why it lives here
 * rather than in a private field of one of them.
 *
 * <p>A pool whose worker threads implement {@link Flagged} answers from a plain field on the
 * thread, which costs a field access rather than a {@code ThreadLocalMap} probe and write. Any
 * other thread -- an application thread that publishes work, or a worker of a pool that predates
 * this interface -- falls back to the thread local. Both are visible to {@link #active()}, so the
 * cross-pool property holds whichever kind of thread is running the callback.
 */
final class GgufCallbackScope {

  /** A thread that can carry the flag itself instead of paying for a thread local. */
  interface Flagged {

    boolean inGgufCallback();

    void inGgufCallback(boolean value);
  }

  private static final ThreadLocal<Boolean> FLAG = ThreadLocal.withInitial(() -> Boolean.FALSE);

  private GgufCallbackScope() {}

  static boolean active() {
    Thread current = Thread.currentThread();
    if (current instanceof Flagged flagged) {
      return flagged.inGgufCallback();
    }
    return FLAG.get();
  }

  static void enter() {
    set(true);
  }

  static void leave() {
    set(false);
  }

  private static void set(boolean value) {
    Thread current = Thread.currentThread();
    if (current instanceof Flagged flagged) {
      flagged.inGgufCallback(value);
      return;
    }
    FLAG.set(value);
  }
}
