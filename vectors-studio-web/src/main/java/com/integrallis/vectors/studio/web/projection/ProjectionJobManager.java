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
package com.integrallis.vectors.studio.web.projection;

import com.integrallis.vectors.studio.core.StudioSession;
import com.integrallis.vectors.studio.core.projection.ProjectionRequest;
import com.integrallis.vectors.studio.core.projection.ProjectionResult;
import java.time.Duration;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/** Tracks outstanding projection jobs; runs each on a virtual thread. */
public final class ProjectionJobManager implements AutoCloseable {

  private static final Duration IDLE_TTL = Duration.ofMinutes(10);

  private final ConcurrentHashMap<String, ProjectionJob> jobs = new ConcurrentHashMap<>();
  private final ScheduledExecutorService janitor =
      Executors.newSingleThreadScheduledExecutor(r -> Thread.ofVirtual().unstarted(r));

  public ProjectionJobManager() {
    janitor.scheduleAtFixedRate(this::expireIdle, 1, 1, TimeUnit.MINUTES);
  }

  /** Submits a job. Returns the assigned id; the projection runs asynchronously. */
  public String submit(StudioSession s, ProjectionRequest req, float[][] data, String[] ids) {
    String jobId = UUID.randomUUID().toString();
    ProjectionJob job = new ProjectionJob(jobId, ids);
    jobs.put(jobId, job);
    Thread.ofVirtual()
        .name("projection-" + jobId)
        .start(
            () -> {
              try {
                ProjectionResult r = s.projectionRunner().run(req, data, job.listener());
                job.complete(r);
              } catch (Throwable t) {
                job.fail(t);
              }
            });
    return jobId;
  }

  public ProjectionJob get(String jobId) {
    return jobs.get(jobId);
  }

  public boolean cancel(String jobId) {
    ProjectionJob j = jobs.get(jobId);
    return j != null && j.cancel();
  }

  private void expireIdle() {
    long cutoff = System.currentTimeMillis() - IDLE_TTL.toMillis();
    jobs.values()
        .removeIf(
            j -> {
              if (j.lastTouched() < cutoff && j.state() != ProjectionJob.State.RUNNING) {
                return true;
              }
              return false;
            });
  }

  @Override
  public void close() {
    janitor.shutdownNow();
    jobs.values().forEach(ProjectionJob::cancel);
    jobs.clear();
  }
}
