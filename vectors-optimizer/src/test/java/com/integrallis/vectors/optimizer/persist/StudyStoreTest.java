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
package com.integrallis.vectors.optimizer.persist;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.optimizer.space.Trial;
import com.integrallis.vectors.optimizer.study.TrialResult;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class StudyStoreTest {

  private static TrialResult sampleResult(String trialId, double score) {
    Map<String, Object> p = new LinkedHashMap<>();
    p.put("m", 16);
    p.put("efConstruction", 200);
    p.put("metric", "COSINE");
    return new TrialResult(
        new Trial(trialId, p),
        Instant.parse("2026-05-09T12:00:00Z"),
        Instant.parse("2026-05-09T12:00:01Z"),
        0.85,
        0.82,
        0.5,
        0.62,
        0.7,
        12.5,
        35.0,
        80.0,
        140L,
        1_048_576L,
        score);
  }

  @Test
  void appendsAndReadsJsonLines(@TempDir Path tmp) {
    StudyStore store = new StudyStore(tmp);
    store.appendTrial("study-A", sampleResult("t-0", 0.9));
    store.appendTrial("study-A", sampleResult("t-1", 0.7));

    var loaded = store.loadAll("study-A");
    assertThat(loaded).hasSize(2);
    assertThat(loaded.get(0).trial().trialId()).isEqualTo("t-0");
    assertThat(loaded.get(0).objectiveScore()).isEqualTo(0.9);
    assertThat(loaded.get(0).trial().getInt("m")).isEqualTo(16);
    assertThat(loaded.get(1).trial().trialId()).isEqualTo("t-1");
  }

  @Test
  void survivesProcessRestart(@TempDir Path tmp) {
    // Simulate restart: write with one StudyStore instance, read with a fresh one.
    new StudyStore(tmp).appendTrial("study-B", sampleResult("t-0", 0.5));
    new StudyStore(tmp).appendTrial("study-B", sampleResult("t-1", 0.6));

    StudyStore reopened = new StudyStore(tmp);
    var loaded = reopened.loadAll("study-B");
    assertThat(loaded).hasSize(2);
    assertThat(loaded.get(1).objectiveScore()).isEqualTo(0.6);

    var summaries = reopened.listStudies();
    assertThat(summaries).hasSize(1);
    assertThat(summaries.get(0).studyId()).isEqualTo("study-B");
    assertThat(summaries.get(0).trialCount()).isEqualTo(2);
  }

  @Test
  void concurrentAppendsSerialised(@TempDir Path tmp) throws Exception {
    StudyStore store = new StudyStore(tmp);
    int writers = 4;
    int perWriter = 25;
    int total = writers * perWriter;
    CountDownLatch start = new CountDownLatch(1);
    ExecutorService pool = Executors.newFixedThreadPool(writers);
    for (int w = 0; w < writers; w++) {
      final int wid = w;
      pool.submit(
          () -> {
            try {
              start.await();
              for (int i = 0; i < perWriter; i++) {
                store.appendTrial("study-C", sampleResult("w" + wid + "-t" + i, wid * 0.1 + i));
              }
            } catch (InterruptedException e) {
              Thread.currentThread().interrupt();
            }
          });
    }
    start.countDown();
    pool.shutdown();
    assertThat(pool.awaitTermination(20, TimeUnit.SECONDS)).isTrue();

    var loaded = store.loadAll("study-C");
    assertThat(loaded).hasSize(total);
  }

  @Test
  void writeMetaSidecar(@TempDir Path tmp) {
    StudyStore store = new StudyStore(tmp);
    store.writeMeta("study-D", Map.of("nTrials", 10, "samplerKind", "TPE"));
    assertThat(tmp.resolve("study-D.meta.json")).exists();
  }
}
