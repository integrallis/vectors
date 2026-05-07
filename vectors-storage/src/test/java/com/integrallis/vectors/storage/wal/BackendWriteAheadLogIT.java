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
package com.integrallis.vectors.storage.wal;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.storage.backend.S3StorageBackend;
import java.io.IOException;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.localstack.LocalStackContainer;
import org.testcontainers.containers.localstack.LocalStackContainer.Service;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.CreateBucketRequest;

/**
 * Integration test for {@link BackendWriteAheadLog} against a real S3 endpoint (LocalStack 3.8).
 * Validates the §6.2 / §16.2 contract end-to-end across S3 PUT/GET so the durability invariants
 * proven by the in-process contract test also hold over the network path.
 *
 * <p>Tagged {@code @Tag("integration")} — requires Docker.
 */
@Tag("integration")
@Testcontainers
class BackendWriteAheadLogIT {

  private static final String BUCKET = "test-vectors-wal";

  @Container
  static final LocalStackContainer LOCALSTACK =
      new LocalStackContainer(DockerImageName.parse("localstack/localstack:3.8"))
          .withServices(Service.S3);

  private static S3Client s3Client;

  @BeforeAll
  static void createBucket() {
    s3Client =
        S3Client.builder()
            .endpointOverride(LOCALSTACK.getEndpoint())
            .region(Region.of(LOCALSTACK.getRegion()))
            .credentialsProvider(
                StaticCredentialsProvider.create(
                    AwsBasicCredentials.create(
                        LOCALSTACK.getAccessKey(), LOCALSTACK.getSecretKey())))
            .forcePathStyle(true)
            .build();
    s3Client.createBucket(CreateBucketRequest.builder().bucket(BUCKET).build());
  }

  @AfterAll
  static void closeClient() {
    if (s3Client != null) s3Client.close();
  }

  private S3StorageBackend backend() {
    return new S3StorageBackend(s3Client, BUCKET);
  }

  private BackendWriteAheadLog wal(S3StorageBackend b, String ns) throws IOException {
    return new BackendWriteAheadLog(b, ns, Duration.ofMillis(100), 512 * 1024 * 1024);
  }

  @Test
  void appendThenReadRoundTripsViaS3() throws IOException {
    String ns = "wal-" + UUID.randomUUID();
    S3StorageBackend b = backend();
    try (BackendWriteAheadLog w = wal(b, ns)) {
      w.append("alpha".getBytes());
      w.append("beta".getBytes());
      try (Stream<WriteAheadLog.WalEntry> s = w.readFrom(0)) {
        assertThat(s.toList())
            .extracting(e -> new String(e.data()))
            .containsExactly("alpha", "beta");
      }
    }
  }

  @Test
  void reopenRecoversManifestFromS3() throws IOException {
    String ns = "wal-" + UUID.randomUUID();
    S3StorageBackend b = backend();
    try (BackendWriteAheadLog w = wal(b, ns)) {
      w.append("a".getBytes());
      w.append("b".getBytes());
    }
    try (BackendWriteAheadLog w = wal(b, ns)) {
      assertThat(w.lastSequenceNumber()).isEqualTo(1L);
      long seq = w.append("c".getBytes());
      assertThat(seq).isEqualTo(2L);
      try (Stream<WriteAheadLog.WalEntry> s = w.readFrom(0)) {
        assertThat(s.toList())
            .extracting(e -> new String(e.data()))
            .containsExactly("a", "b", "c");
      }
    }
  }

  @Test
  void groupCommitCoalescesConcurrentAppendsIntoOnePut() throws Exception {
    String ns = "wal-" + UUID.randomUUID();
    S3StorageBackend b = backend();
    int n = 200;
    try (BackendWriteAheadLog w =
        new BackendWriteAheadLog(b, ns, Duration.ofMillis(250), 512 * 1024 * 1024)) {
      CountDownLatch ready = new CountDownLatch(n);
      CountDownLatch go = new CountDownLatch(1);
      try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
        for (int i = 0; i < n; i++) {
          final int id = i;
          pool.submit(
              () -> {
                ready.countDown();
                try {
                  go.await();
                  w.append(("entry-" + id).getBytes());
                } catch (Exception e) {
                  throw new RuntimeException(e);
                }
              });
        }
        ready.await(5, TimeUnit.SECONDS);
        go.countDown();
      }
      assertThat(w.lastSequenceNumber()).isEqualTo(n - 1L);
      assertThat(w.putCount()).isEqualTo(1L);
      List<String> walSegs = b.list(ns + "/wal/");
      assertThat(walSegs).hasSize(1);
    }
  }
}
