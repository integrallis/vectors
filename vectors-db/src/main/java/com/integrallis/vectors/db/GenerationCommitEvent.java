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

import java.nio.file.Path;

/**
 * Describes a generation that has just been committed and made durable + visible, delivered to a
 * {@link GenerationSubscriber} (P3.1 read-replica directory shipping).
 *
 * <p>At delivery time the entire {@code gen-NNNN/} directory is on disk, CRC-valid, and the {@code
 * CURRENT} pointer references it. A subscriber may read every file under {@link #generationDir()}
 * safely (the producer never rewrites a finalized generation directory).
 *
 * @param generationNumber the committed generation number
 * @param generationDir absolute path to the finalized {@code gen-NNNN/} directory
 * @param storageRoot absolute path to the collection's storage root (parent of {@code
 *     generationDir} and the {@code CURRENT} pointer)
 */
public record GenerationCommitEvent(long generationNumber, Path generationDir, Path storageRoot) {}
