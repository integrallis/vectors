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
import com.integrallis.vectors.studio.core.projection.ProgressListener;
import com.integrallis.vectors.studio.core.projection.ProjectionAlgorithm;
import com.integrallis.vectors.studio.core.projection.ProjectionRequest;
import com.integrallis.vectors.studio.core.projection.ProjectionResult;
import com.integrallis.vectors.studio.core.projection.QueryProjector;
import java.time.Duration;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/** Tracks outstanding projection jobs and runs CPU-bound projections on a bounded worker pool. */
public final class ProjectionJobManager implements AutoCloseable {

  private static final Duration IDLE_TTL = Duration.ofMinutes(10);
  private static final Duration CLOSE_TIMEOUT = Duration.ofSeconds(5);
  private static final int DEFAULT_MAX_WORKERS =
      Math.max(1, Runtime.getRuntime().availableProcessors() - 1);

  private final ConcurrentHashMap<String, ProjectionJob> jobs = new ConcurrentHashMap<>();
  private final ConcurrentHashMap<String, RunningProjection> running = new ConcurrentHashMap<>();
  // Latest completed projection per collection, retained so a subsequent text search can place the
  // query as its own point in the same coordinate frame. A new projection overwrites the previous.
  private final ConcurrentHashMap<String, CachedProjection> queryProjections =
      new ConcurrentHashMap<>();
  private final ThreadPoolExecutor workers;
  private final ScheduledExecutorService janitor =
      Executors.newSingleThreadScheduledExecutor(
          r -> {
            Thread t = new Thread(r, "projection-janitor");
            t.setDaemon(true);
            return t;
          });
  private final Duration idleTtl;
  private final AtomicBoolean closed = new AtomicBoolean();

  public ProjectionJobManager() {
    this(DEFAULT_MAX_WORKERS, DEFAULT_MAX_WORKERS, IDLE_TTL);
  }

  ProjectionJobManager(int maxWorkers, int maxQueued, Duration idleTtl) {
    if (maxWorkers < 1) {
      throw new IllegalArgumentException("maxWorkers must be >= 1: " + maxWorkers);
    }
    if (maxQueued < 0) {
      throw new IllegalArgumentException("maxQueued must be >= 0: " + maxQueued);
    }
    this.idleTtl = Objects.requireNonNull(idleTtl, "idleTtl");
    this.workers =
        new ThreadPoolExecutor(
            maxWorkers,
            maxWorkers,
            0L,
            TimeUnit.MILLISECONDS,
            maxQueued == 0 ? new SynchronousQueue<>() : new ArrayBlockingQueue<>(maxQueued),
            r -> {
              Thread t = new Thread(r, "projection-worker");
              t.setDaemon(true);
              return t;
            },
            new ThreadPoolExecutor.AbortPolicy());
    janitor.scheduleAtFixedRate(this::expireIdle, 1, 1, TimeUnit.MINUTES);
  }

  /**
   * A completed projection retained per collection so a later query can be projected into its
   * space.
   *
   * @param projector out-of-sample projector (PCA); {@code null} for t-SNE/UMAP (approximate
   *     client-side)
   * @param sphereize whether the submitted data was L2-normalized before fitting; the query must be
   *     normalized the same way before projecting
   * @param algorithm the algorithm that produced this projection
   */
  public record CachedProjection(
      QueryProjector projector, boolean sphereize, ProjectionAlgorithm algorithm) {}

  /** Submits a job. Returns the assigned id; the projection runs asynchronously. */
  public String submit(
      StudioSession s, ProjectionRequest req, float[][] data, String[] ids, boolean sphereize) {
    Objects.requireNonNull(s, "s");
    Objects.requireNonNull(req, "req");
    Objects.requireNonNull(data, "data");
    Objects.requireNonNull(ids, "ids");
    String collection = req.collection();
    ProjectionAlgorithm algorithm = req.algorithm();
    return submit(
        ids,
        listener -> {
          ProjectionResult r = s.projectionRunner().run(req, data, listener);
          queryProjections.put(
              collection, new CachedProjection(r.queryProjector(), sphereize, algorithm));
          return r;
        });
  }

  /** The most recent completed projection for {@code collection}, if any. */
  public Optional<CachedProjection> queryProjectionFor(String collection) {
    return Optional.ofNullable(queryProjections.get(collection));
  }

  String submit(String[] ids, ProjectionTask task) {
    Objects.requireNonNull(ids, "ids");
    Objects.requireNonNull(task, "task");
    if (closed.get()) {
      throw new RejectedExecutionException("projection job manager is closed");
    }
    String jobId = UUID.randomUUID().toString();
    ProjectionJob job = new ProjectionJob(jobId, ids);
    RunningProjection runningJob = new RunningProjection(job);
    jobs.put(jobId, job);
    running.put(jobId, runningJob);
    try {
      Future<?> future =
          workers.submit(
              () -> {
                Thread current = Thread.currentThread();
                String previousName = current.getName();
                current.setName("projection-" + jobId);
                runningJob.thread.set(current);
                try {
                  ProjectionResult r = task.run(job.listener());
                  job.complete(r);
                } catch (CancellationException e) {
                  job.cancel();
                } catch (Throwable t) {
                  if (job.state() == ProjectionJob.State.CANCELLED
                      || Thread.currentThread().isInterrupted()) {
                    job.cancel();
                  } else {
                    job.fail(t);
                  }
                } finally {
                  running.remove(jobId);
                  runningJob.thread.set(null);
                  current.setName(previousName);
                }
              });
      runningJob.setFuture(future);
    } catch (RejectedExecutionException e) {
      running.remove(jobId);
      jobs.remove(jobId);
      job.fail(e);
      throw e;
    }
    return jobId;
  }

  @FunctionalInterface
  interface ProjectionTask {
    ProjectionResult run(ProgressListener listener) throws Exception;
  }

  private static final class RunningProjection {
    private final ProjectionJob job;
    private final AtomicReference<Thread> thread = new AtomicReference<>();
    private final AtomicReference<Future<?>> future = new AtomicReference<>();

    private RunningProjection(ProjectionJob job) {
      this.job = job;
    }

    private void setFuture(Future<?> f) {
      future.set(f);
      if (job.state() == ProjectionJob.State.CANCELLED) {
        f.cancel(true);
      }
    }

    private void interruptAndCancel() {
      Thread t = thread.get();
      if (t != null) {
        t.interrupt();
      }
      Future<?> f = future.get();
      if (f != null) {
        f.cancel(true);
      }
    }
  }

  public ProjectionJob get(String jobId) {
    return jobs.get(jobId);
  }

  public boolean cancel(String jobId) {
    ProjectionJob j = jobs.get(jobId);
    if (j == null) {
      return false;
    }
    boolean cancelled = j.cancel();
    RunningProjection runningJob = running.get(jobId);
    if (runningJob != null) {
      runningJob.interruptAndCancel();
    }
    return cancelled;
  }

  void expireIdle() {
    long cutoff = System.currentTimeMillis() - idleTtl.toMillis();
    jobs.forEach(
        (id, job) -> {
          if (job.lastTouched() >= cutoff) {
            return;
          }
          if (job.state() == ProjectionJob.State.RUNNING) {
            cancel(id);
          }
          if (job.state() != ProjectionJob.State.RUNNING) {
            jobs.remove(id, job);
          }
        });
  }

  @Override
  public void close() {
    if (!closed.compareAndSet(false, true)) {
      return;
    }
    janitor.shutdownNow();
    jobs.keySet().forEach(this::cancel);
    workers.shutdownNow();
    try {
      if (!workers.awaitTermination(CLOSE_TIMEOUT.toMillis(), TimeUnit.MILLISECONDS)) {
        running.values().forEach(RunningProjection::interruptAndCancel);
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      running.values().forEach(RunningProjection::interruptAndCancel);
    } finally {
      jobs.clear();
      running.clear();
    }
  }
}
