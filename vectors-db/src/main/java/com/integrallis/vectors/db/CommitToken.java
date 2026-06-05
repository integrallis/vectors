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
 * Handle to a completed ingest: the generation that the documents landed in and how many documents
 * were committed. Returned (via a {@link java.util.concurrent.CompletionStage}) by {@link
 * AsyncBatchIngestor} when an async ingest finishes (I.8).
 *
 * @param generationNumber the committed generation number after the final commit ({@code -1} if the
 *     collection does not track generations), e.g. usable with {@link VectorCollection#refresh()}
 *     on a replica
 * @param documentsCommitted the number of documents ingested and committed by this ingest
 */
public record CommitToken(long generationNumber, long documentsCommitted) {}
