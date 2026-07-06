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
package com.integrallis.vectors.studio.web.dataset;

import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.studio.core.StudioSession;
import com.integrallis.vectors.studio.core.dataset.DatasetLoader;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Runs sample-dataset loads on a background thread pool, tracking one {@link DatasetLoadJob} per
 * submission and publishing progress as rows are staged. Mirrors the async-job-with-SSE pattern
 * used by {@code ProjectionJobManager}, minus cancellation (a load is short and network-bound). On
 * success the freshly built collection is registered into the session backend under the job's
 * collection name.
 */
public final class DatasetLoadJobManager implements AutoCloseable {

  private static final Logger LOG = LoggerFactory.getLogger(DatasetLoadJobManager.class);
  private static final int MAX_WORKERS = 2;

  private final ConcurrentHashMap<String, DatasetLoadJob> jobs = new ConcurrentHashMap<>();
  private final ExecutorService workers =
      Executors.newFixedThreadPool(
          MAX_WORKERS,
          r -> {
            Thread t = new Thread(r, "dataset-loader");
            t.setDaemon(true);
            return t;
          });
  private final AtomicBoolean closed = new AtomicBoolean();

  /**
   * Submits a load. The job runs asynchronously: it pages rows through {@code fetcher}, builds the
   * collection, and on success registers it under {@code collectionName} in the session backend.
   *
   * @return the assigned job id
   */
  public String submit(
      StudioSession session,
      String collectionName,
      DatasetLoader.Config cfg,
      DatasetLoader.RowsFetcher fetcher,
      int total) {
    if (closed.get()) {
      throw new IllegalStateException("dataset job manager is closed");
    }
    String jobId = UUID.randomUUID().toString();
    DatasetLoadJob job = new DatasetLoadJob(jobId, collectionName, total);
    jobs.put(jobId, job);
    workers.submit(
        () -> {
          try {
            VectorCollection collection = DatasetLoader.load(cfg, fetcher, job::progress);
            session.backend().addCollection(collectionName, collection);
            job.complete(collection.size());
          } catch (Throwable t) {
            LOG.warn("studio: dataset load '{}' failed: {}", collectionName, t.getMessage());
            job.fail(t.getMessage() == null ? t.getClass().getSimpleName() : t.getMessage());
          }
        });
    return jobId;
  }

  public DatasetLoadJob get(String jobId) {
    return jobs.get(jobId);
  }

  @Override
  public void close() {
    if (closed.compareAndSet(false, true)) {
      workers.shutdownNow();
    }
  }
}
