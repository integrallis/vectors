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
package com.integrallis.vectors.storage.io;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utility for calling {@code posix_madvise()} on memory-mapped segments via FFM. Silently falls
 * back to no-op on platforms that don't support it (Windows).
 */
final class MadviseUtil {

  private static final Logger log = LoggerFactory.getLogger(MadviseUtil.class);

  // POSIX madvise constants (same on Linux and macOS)
  private static final int POSIX_MADV_NORMAL = 0;
  private static final int POSIX_MADV_RANDOM = 1;
  private static final int POSIX_MADV_SEQUENTIAL = 2;
  private static final int POSIX_MADV_WILLNEED = 3;

  private static final MethodHandle MADVISE_HANDLE = initMadviseHandle();

  /**
   * Set once when the first apply() call observes a runtime failure, so the warning fires exactly
   * once per JVM instead of spamming the log on every mmap.
   */
  private static final java.util.concurrent.atomic.AtomicBoolean APPLY_FAILURE_LOGGED =
      new java.util.concurrent.atomic.AtomicBoolean();

  private MadviseUtil() {}

  @SuppressWarnings("restricted")
  private static MethodHandle initMadviseHandle() {
    try {
      Linker linker = Linker.nativeLinker();
      var lookup = linker.defaultLookup();
      var symbol = lookup.find("posix_madvise");
      if (symbol.isPresent()) {
        return linker.downcallHandle(
            symbol.get(),
            FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_LONG,
                ValueLayout.JAVA_INT));
      }
    } catch (Exception e) {
      log.debug("posix_madvise not available on this platform: {}", e.getMessage());
    }
    return null;
  }

  /** Returns true if posix_madvise is available on this platform. */
  static boolean isAvailable() {
    return MADVISE_HANDLE != null;
  }

  /**
   * Applies the given madvise strategy to a memory-mapped segment.
   *
   * @param segment the mapped segment to advise
   * @param strategy the advisory strategy
   */
  static void apply(MemorySegment segment, MadviseStrategy strategy) {
    if (strategy == MadviseStrategy.NONE || MADVISE_HANDLE == null) {
      return;
    }
    int advice = toNativeAdvice(strategy);
    try {
      int result = invokeMadvise(segment, segment.byteSize(), advice);
      if (result != 0) {
        // First non-zero result is surfaced once at WARN; subsequent ones drop to DEBUG so a
        // platform that silently rejects every madvise doesn't spam the log. Audit recommended
        // surfacing the "no-op madvise" case so it's not invisible.
        if (APPLY_FAILURE_LOGGED.compareAndSet(false, true)) {
          log.warn(
              "posix_madvise returned non-zero: {} for strategy {} (further failures at debug)",
              result,
              strategy);
        } else {
          log.debug("posix_madvise returned non-zero: {} for strategy {}", result, strategy);
        }
      }
    } catch (Throwable t) {
      if (APPLY_FAILURE_LOGGED.compareAndSet(false, true)) {
        log.warn(
            "Failed to apply madvise strategy {} (further failures at debug): {}",
            strategy,
            t.getMessage());
      } else {
        log.debug("Failed to apply madvise strategy {}: {}", strategy, t.getMessage());
      }
    }
  }

  @SuppressWarnings("restricted")
  private static int invokeMadvise(MemorySegment segment, long size, int advice) throws Throwable {
    return (int) MADVISE_HANDLE.invokeExact(segment, size, advice);
  }

  private static int toNativeAdvice(MadviseStrategy strategy) {
    return switch (strategy) {
      case RANDOM -> POSIX_MADV_RANDOM;
      case SEQUENTIAL -> POSIX_MADV_SEQUENTIAL;
      case WILLNEED -> POSIX_MADV_WILLNEED;
      case NONE -> POSIX_MADV_NORMAL;
    };
  }
}
