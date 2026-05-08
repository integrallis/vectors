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

import com.integrallis.vectors.storage.backend.S3StorageBackend;
import com.integrallis.vectors.storage.wal.BackendWriteAheadLog;
import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.util.Random;
import java.util.UUID;
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
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;

/**
 * JMH benchmark for {@link BackendWriteAheadLog} over a real S3 endpoint. Measures the §16.2 gates
 * from {@code vectors-distributed-design.md}: {@code p50 ≤ 285 ms} for a 500 kB append and {@code
 * 10 k appends/s/namespace} sustained throughput.
 *
 * <p>Excluded from the default {@code :vectors-bench:jmh} run because it requires an external S3
 * endpoint. Opt in by setting:
 *
 * <pre>{@code
 * # against LocalStack:
 * docker run --rm -d -p 4566:4566 localstack/localstack:3.8
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=WriteAheadLogS3Benchmark \
 *     -Pwal.s3.endpoint=http://localhost:4566 \
 *     -Pwal.s3.bucket=test-wal -Pwal.s3.region=us-east-1
 *
 * # against real S3 (uses default credential chain):
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=WriteAheadLogS3Benchmark \
 *     -Pwal.s3.bucket=my-bucket -Pwal.s3.region=us-west-2
 * }</pre>
 */
@BenchmarkMode({Mode.SampleTime, Mode.Throughput})
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 2, time = 2)
@Measurement(iterations = 5, time = 3)
public class WriteAheadLogS3Benchmark {

  @Param({"512000"})
  public int payloadBytes;

  @Param({"1000"})
  public int groupCommitMillis;

  private S3Client s3;
  private S3StorageBackend backend;
  private BackendWriteAheadLog wal;
  private byte[] payload;
  private String namespace;

  @Setup(Level.Trial)
  public void setUp() throws IOException {
    String bucket = System.getProperty("wal.s3.bucket");
    String region = System.getProperty("wal.s3.region", "us-east-1");
    String endpoint = System.getProperty("wal.s3.endpoint");
    if (bucket == null) {
      throw new IllegalStateException(
          "WriteAheadLogS3Benchmark requires -Dwal.s3.bucket=... (and optionally -Dwal.s3.endpoint, -Dwal.s3.region)");
    }
    var b = S3Client.builder().region(Region.of(region));
    if (endpoint != null) {
      // LocalStack / MinIO path: static creds + path-style addressing.
      b =
          b.endpointOverride(URI.create(endpoint))
              .forcePathStyle(true)
              .credentialsProvider(
                  StaticCredentialsProvider.create(AwsBasicCredentials.create("test", "test")));
    } else {
      b = b.credentialsProvider(DefaultCredentialsProvider.create());
    }
    s3 = b.build();
    backend = new S3StorageBackend(s3, bucket);
    namespace = "wal-bench-" + UUID.randomUUID();
    wal =
        new BackendWriteAheadLog(
            backend, namespace, Duration.ofMillis(groupCommitMillis), 512 * 1024 * 1024);
    payload = new byte[payloadBytes];
    new Random(42L).nextBytes(payload);
  }

  @TearDown(Level.Trial)
  public void tearDown() throws IOException {
    if (wal != null) wal.close();
    if (backend != null) {
      // Best-effort cleanup of objects under the benchmark namespace.
      for (String key : backend.list(namespace + "/")) {
        try {
          backend.delete(key);
        } catch (IOException ignored) {
        }
      }
    }
    if (s3 != null) s3.close();
  }

  @Benchmark
  @Threads(1)
  public long append_500KB_singleThread() throws IOException {
    return wal.append(payload);
  }

  @Benchmark
  @Threads(32)
  public long append_500KB_throughput32() throws IOException {
    return wal.append(payload);
  }
}
