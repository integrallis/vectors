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
package com.integrallis.vectors.bench;

import com.integrallis.vectors.storage.backend.LocalFileStorageBackend;
import com.integrallis.vectors.storage.wal.BackendWriteAheadLog;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;

/**
 * JMH benchmark for {@link BackendWriteAheadLog} group-commit latency and throughput against a
 * local filesystem backend.
 *
 * <p>Tracks the §6.2 / §16.2 contract from {@code vectors-distributed-design.md}. The local-backend
 * numbers are an upper bound on what the WAL framing + group-commit machinery can sustain when
 * object-PUT latency is near zero — they isolate the implementation overhead from any S3-specific
 * cost.
 *
 * <p>Run:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=WriteAheadLogBenchmark
 * }</pre>
 *
 * <p>For S3 numbers see {@code WriteAheadLogS3Benchmark} (opt-in via {@code
 * -Pwal.s3.endpoint=...}).
 */
@BenchmarkMode({Mode.SampleTime, Mode.Throughput})
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 2, time = 2)
@Measurement(iterations = 5, time = 3)
public class WriteAheadLogBenchmark {

  @Param({"512000"})
  public int payloadBytes;

  @Param({"1000"})
  public int groupCommitMillis;

  private Path tmpDir;
  private LocalFileStorageBackend backend;
  private BackendWriteAheadLog wal;
  private byte[] payload;

  @Setup(Level.Trial)
  public void setUp() throws IOException {
    tmpDir = Files.createTempDirectory("wal-bench-");
    backend = new LocalFileStorageBackend(tmpDir);
    wal =
        new BackendWriteAheadLog(
            backend, "bench", Duration.ofMillis(groupCommitMillis), 512 * 1024 * 1024);
    payload = new byte[payloadBytes];
    new Random(42L).nextBytes(payload);
  }

  @TearDown(Level.Trial)
  public void tearDown() throws IOException {
    if (wal != null) wal.close();
    if (tmpDir != null) {
      try (var paths = Files.walk(tmpDir)) {
        paths
            .sorted((a, b) -> b.getNameCount() - a.getNameCount())
            .forEach(
                p -> {
                  try {
                    Files.deleteIfExists(p);
                  } catch (IOException ignored) {
                  }
                });
      }
    }
  }

  /**
   * Single-threaded append latency. With a 1 s group-commit interval the first append in each
   * window pays the full commit-cycle wait; subsequent appends within the window coalesce. JMH
   * {@link Mode#SampleTime} reports p50/p90/p99 directly so the §16.2 {@code p50 ≤ 285 ms} latency
   * gate can be read from the histogram.
   */
  @Benchmark
  @Threads(1)
  public long append_500KB_singleThread() throws IOException {
    return wal.append(payload);
  }

  /**
   * 32-thread concurrent throughput. The whole point of group commit is that a fan-in of writers
   * coalesces into one PUT per commit window, so this measures the §16.2 {@code 10 k appends /s
   * /namespace} throughput target rather than per-call latency.
   */
  @Benchmark
  @Threads(32)
  public long append_500KB_throughput32() throws IOException {
    return wal.append(payload);
  }
}
