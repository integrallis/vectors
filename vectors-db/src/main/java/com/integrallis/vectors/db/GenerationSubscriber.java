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
package com.integrallis.vectors.db;

/**
 * Hook invoked after a persistent collection commits a new generation (P3.1 read-replica directory
 * shipping). The canonical implementation, {@link GenerationShippingSubscriber}, ships the
 * generation directory to a remote {@code StorageBackend} so read-replicas can pick it up via
 * {@link VectorCollection#refresh()}.
 *
 * <p><b>Threading.</b> {@link #onGenerationCommitted} is called by the committing thread while the
 * collection's writer lock is held, in generation order. Implementations <b>must not block</b> — do
 * any I/O asynchronously (the shipping implementation enqueues to a background executor and returns
 * immediately). A subscriber that throws is logged and skipped; it never fails the commit.
 *
 * <p><b>Lifecycle.</b> If an implementation also implements {@link AutoCloseable}, the owning
 * collection closes it from {@link VectorCollection#close()}.
 */
@FunctionalInterface
public interface GenerationSubscriber {

  /**
   * Called once per committed generation, in order, on the committing thread under the writer lock.
   * Must return promptly (no blocking I/O).
   *
   * @param event the committed generation (directory is on disk, CRC-valid, and current)
   */
  void onGenerationCommitted(GenerationCommitEvent event);
}
