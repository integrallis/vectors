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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.studio.core.projection.ProjectionAlgorithm;
import com.integrallis.vectors.studio.core.projection.ProjectionParams;
import com.integrallis.vectors.studio.core.projection.ProjectionResult;
import java.time.Duration;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ProjectionJobManagerTest {

  @Test
  void cancelInterruptsRunningProjection() throws Exception {
    ProjectionJobManager mgr = new ProjectionJobManager(1, 1, Duration.ofMinutes(1));
    CountDownLatch started = new CountDownLatch(1);
    CountDownLatch interrupted = new CountDownLatch(1);

    String jobId =
        mgr.submit(
            new String[] {"doc-1"},
            listener -> {
              started.countDown();
              try {
                Thread.sleep(10_000);
              } catch (InterruptedException e) {
                interrupted.countDown();
                throw new CancellationException("interrupted");
              }
              return result();
            });

    assertThat(started.await(2, TimeUnit.SECONDS)).isTrue();
    assertThat(mgr.cancel(jobId)).isTrue();

    assertThat(interrupted.await(2, TimeUnit.SECONDS)).isTrue();
    assertThat(mgr.get(jobId).state()).isEqualTo(ProjectionJob.State.CANCELLED);
    mgr.close();
  }

  @Test
  void closeInterruptsRunningProjectionAndClearsJobs() throws Exception {
    ProjectionJobManager mgr = new ProjectionJobManager(1, 1, Duration.ofMinutes(1));
    CountDownLatch started = new CountDownLatch(1);
    CountDownLatch interrupted = new CountDownLatch(1);

    String jobId =
        mgr.submit(
            new String[] {"doc-1"},
            listener -> {
              started.countDown();
              try {
                Thread.sleep(10_000);
              } catch (InterruptedException e) {
                interrupted.countDown();
                throw new CancellationException("interrupted");
              }
              return result();
            });

    assertThat(started.await(2, TimeUnit.SECONDS)).isTrue();
    mgr.close();

    assertThat(interrupted.await(2, TimeUnit.SECONDS)).isTrue();
    assertThat(mgr.get(jobId)).isNull();
  }

  @Test
  void saturatedManagerRejectsInsteadOfQueueingUnboundedMatrices() throws Exception {
    ProjectionJobManager mgr = new ProjectionJobManager(1, 0, Duration.ofMinutes(1));
    CountDownLatch started = new CountDownLatch(1);
    CountDownLatch release = new CountDownLatch(1);

    mgr.submit(
        new String[] {"doc-1"},
        listener -> {
          started.countDown();
          release.await();
          return result();
        });
    assertThat(started.await(2, TimeUnit.SECONDS)).isTrue();

    assertThatThrownBy(() -> mgr.submit(new String[] {"doc-2"}, listener -> result()))
        .isInstanceOf(RejectedExecutionException.class);

    release.countDown();
    mgr.close();
  }

  private static ProjectionResult result() {
    return new ProjectionResult(
        new float[][] {{0f, 0f}},
        ProjectionAlgorithm.PCA,
        new ProjectionParams.PcaParams(2, true, false),
        0L,
        null,
        null);
  }
}
