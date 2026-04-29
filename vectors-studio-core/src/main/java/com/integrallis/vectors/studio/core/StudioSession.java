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
package com.integrallis.vectors.studio.core;

import com.integrallis.vectors.studio.core.connection.StudioBackend;
import com.integrallis.vectors.studio.core.metadata.MetadataSchema;
import com.integrallis.vectors.studio.core.projection.ProjectionRunner;
import com.integrallis.vectors.studio.core.recommender.HeuristicRecommender;
import com.integrallis.vectors.studio.core.recommender.LlmRecommender;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Per-process Studio session bundling the backend, metadata schemas, recommender, and projection
 * runner.
 */
public final class StudioSession implements AutoCloseable {

  private final StudioBackend backend;
  private final ConcurrentHashMap<String, MetadataSchema> schemas = new ConcurrentHashMap<>();
  private final HeuristicRecommender heuristic = new HeuristicRecommender();
  private final Optional<LlmRecommender> llm;
  private final ProjectionRunner runner = new ProjectionRunner();

  public StudioSession(StudioBackend backend) {
    this(backend, Optional.empty());
  }

  public StudioSession(StudioBackend backend, Optional<LlmRecommender> llm) {
    this.backend = backend;
    this.llm = llm;
  }

  public StudioBackend backend() {
    return backend;
  }

  public MetadataSchema metadataSchema(String collection) {
    return schemas.getOrDefault(collection, MetadataSchema.empty());
  }

  public void putMetadataSchema(String collection, MetadataSchema schema) {
    schemas.put(collection, schema);
  }

  public HeuristicRecommender heuristic() {
    return heuristic;
  }

  public Optional<LlmRecommender> llm() {
    return llm;
  }

  public ProjectionRunner projectionRunner() {
    return runner;
  }

  @Override
  public void close() {
    backend.close();
  }
}
