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
package com.integrallis.vectors.studio.web.optimize;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.optimizer.data.Qrels;
import com.integrallis.vectors.optimizer.data.Queries;
import com.integrallis.vectors.optimizer.embed.EmbeddingProvider;
import com.integrallis.vectors.optimizer.objective.ObjectiveWeights;
import com.integrallis.vectors.optimizer.persist.StudyStore;
import com.integrallis.vectors.optimizer.space.ParamSpec;
import com.integrallis.vectors.optimizer.space.SearchSpace;
import com.integrallis.vectors.optimizer.study.StudyConfig;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.RejectedExecutionException;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class OptimizeJobManagerTest {

  private static final int DIM = 4;

  private static List<Document> corpus() {
    List<Document> out = new ArrayList<>();
    for (int c = 0; c < 4; c++) {
      for (int i = 0; i < 25; i++) {
        float[] v = new float[DIM];
        v[c] = 1.0f;
        for (int d = 0; d < DIM; d++) v[d] += (float) (i * 1e-3);
        out.add(Document.of("doc-" + c + "-" + i, v));
      }
    }
    return out;
  }

  private static StudyConfig fixedConfig() {
    Map<String, String> queryById = new LinkedHashMap<>();
    queryById.put("q-0", "doc-0-0");
    queryById.put("q-1", "doc-1-0");
    queryById.put("q-2", "doc-2-0");
    Map<String, Map<String, Integer>> rel = new LinkedHashMap<>();
    rel.put("q-0", Map.of("doc-0-1", 1, "doc-0-2", 1, "doc-0-3", 1));
    rel.put("q-1", Map.of("doc-1-1", 1, "doc-1-2", 1, "doc-1-3", 1));
    rel.put("q-2", Map.of("doc-2-1", 1, "doc-2-2", 1, "doc-2-3", 1));

    return StudyConfig.builder()
        .searchSpace(
            new SearchSpace(
                List.of(
                    new ParamSpec.FixedString("metric", "COSINE"),
                    new ParamSpec.FixedString("indexType", "FLAT"))))
        .objectiveWeights(ObjectiveWeights.builder().recallWeight(1.0).build())
        .samplerKind(StudyConfig.SamplerKind.RANDOM)
        .corpusSource(OptimizeJobManagerTest::corpus)
        .qrelsSource(() -> new Qrels(rel))
        .queriesSource(() -> new Queries(queryById))
        .nTrials(3)
        .kForMetrics(3)
        .seed(0L)
        .warmupRounds(1)
        .measurementRounds(1)
        .build();
  }

  private static EmbeddingProvider embedder() {
    return text -> {
      // text is "doc-c-i"; embed as the c-th unit vector.
      String[] parts = text.split("-");
      int c = Integer.parseInt(parts[1]);
      float[] v = new float[DIM];
      v[c] = 1.0f;
      return v;
    };
  }

  private static void waitFor(java.util.function.BooleanSupplier cond, Duration max) {
    long deadline = System.nanoTime() + max.toNanos();
    while (System.nanoTime() < deadline) {
      if (cond.getAsBoolean()) return;
      try {
        Thread.sleep(50);
      } catch (InterruptedException ie) {
        Thread.currentThread().interrupt();
        return;
      }
    }
    throw new AssertionError("timed out waiting for condition");
  }

  @Test
  void submittedStudyCompletesAndPersists(@TempDir Path tmp) {
    StudyStore store = new StudyStore(tmp);
    OptimizeJobManager mgr = new OptimizeJobManager(store, Duration.ofMinutes(5));
    String studyId = mgr.submit(fixedConfig(), embedder());

    waitFor(() -> mgr.get(studyId).state() == OptimizeJob.State.COMPLETED, Duration.ofSeconds(20));

    OptimizeJob job = mgr.get(studyId);
    assertThat(job.history()).hasSize(3);
    assertThat(job.history().get(0).objectiveScore()).isFinite();
    // StudyStore should have persisted the same trials.
    assertThat(store.loadAll(studyId)).hasSize(3);
    mgr.close();
  }

  @Test
  void cancelMidRunHaltsBeforeBudgetExhausted(@TempDir Path tmp) {
    StudyStore store = new StudyStore(tmp);
    StudyConfig cfg =
        StudyConfig.builder()
            .searchSpace(
                new SearchSpace(
                    List.of(
                        new ParamSpec.FixedString("metric", "COSINE"),
                        new ParamSpec.FixedString("indexType", "FLAT"))))
            .objectiveWeights(ObjectiveWeights.builder().recallWeight(1.0).build())
            .samplerKind(StudyConfig.SamplerKind.RANDOM)
            .corpusSource(OptimizeJobManagerTest::corpus)
            .qrelsSource(() -> new Qrels(Map.of("q-0", Map.of("doc-0-1", 1))))
            .queriesSource(() -> new Queries(Map.of("q-0", "doc-0-0")))
            .nTrials(500)
            .kForMetrics(3)
            .seed(0L)
            .warmupRounds(2)
            .measurementRounds(3)
            .build();
    OptimizeJobManager mgr = new OptimizeJobManager(store, Duration.ofMinutes(5));
    String studyId = mgr.submit(cfg, embedder());

    // Let the runner complete a couple of trials, then cancel before the (large) budget runs out.
    waitFor(() -> mgr.get(studyId).history().size() >= 1, Duration.ofSeconds(10));
    boolean ok = mgr.cancel(studyId);
    assertThat(ok).isTrue();
    waitFor(
        () -> mgr.get(studyId).state() == OptimizeJob.State.CANCELLED, Duration.ofSeconds(15));
    assertThat(mgr.get(studyId).history().size()).isLessThan(500);
    mgr.close();
  }

  @Test
  void idleTtlEvictsCompletedJobs(@TempDir Path tmp) {
    StudyStore store = new StudyStore(tmp);
    // Tiny TTL so a brief sleep crosses the cutoff; expireIdle() is invoked
    // directly to avoid waiting on the 1-minute janitor cadence.
    OptimizeJobManager mgr = new OptimizeJobManager(store, Duration.ofMillis(50));
    String studyId = mgr.submit(fixedConfig(), embedder());
    waitFor(() -> mgr.get(studyId).state() == OptimizeJob.State.COMPLETED, Duration.ofSeconds(20));

    try {
      Thread.sleep(120);
    } catch (InterruptedException ie) {
      Thread.currentThread().interrupt();
    }
    mgr.expireIdle();

    assertThat(mgr.get(studyId)).isNull();
    mgr.close();
  }

  @Test
  void closeShutsExecutorCleanly(@TempDir Path tmp) {
    StudyStore store = new StudyStore(tmp);
    OptimizeJobManager mgr = new OptimizeJobManager(store, Duration.ofMinutes(5));
    String studyId = mgr.submit(fixedConfig(), embedder());
    waitFor(() -> mgr.get(studyId).state() == OptimizeJob.State.COMPLETED, Duration.ofSeconds(20));

    mgr.close();

    // After close, the jobs map is cleared and the executor refuses new work.
    assertThat(mgr.get(studyId)).isNull();
    org.assertj.core.api.Assertions.assertThatThrownBy(() -> mgr.submit(fixedConfig(), embedder()))
        .isInstanceOf(RejectedExecutionException.class);
  }
}
