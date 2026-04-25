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

/**
 * Memory-mapped I/O advisory strategies. Applied via {@code posix_madvise()} on platforms that
 * support it (Linux, macOS). Silently ignored on unsupported platforms (Windows).
 */
public enum MadviseStrategy {

  /** Random access pattern. Best for ANN search with random graph traversal. Default strategy. */
  RANDOM,

  /** Sequential access pattern. Best for bulk writes, full scans, and compaction. */
  SEQUENTIAL,

  /** Advise the OS to preload pages. Best for hot data like upper HNSW layers. */
  WILLNEED,

  /** No advisory. Let the OS use its default page replacement policy. */
  NONE
}
