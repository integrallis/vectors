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
package com.integrallis.vectors.storage.backend;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;
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
import software.amazon.awssdk.core.client.config.ClientOverrideConfiguration;
import software.amazon.awssdk.core.interceptor.Context;
import software.amazon.awssdk.core.interceptor.ExecutionAttributes;
import software.amazon.awssdk.core.interceptor.ExecutionInterceptor;
import software.amazon.awssdk.http.SdkHttpRequest;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.CreateBucketRequest;

/**
 * Integration test for {@link S3StorageBackend} against LocalStack 3.8.
 *
 * <p>Exercises the same surface that {@link StorageBackendContractTest} covers for the in-JVM
 * backends (put/get/getRange/list/delete/conditionalPut) and additionally verifies that {@link
 * S3StorageBackend#getRange} emits an HTTP {@code Range} header so only the requested bytes
 * traverse the network — the foundational guarantee for the per-ordinal T3 reads in §8.7 and the
 * sub-segment WAL reads in §6.2 of {@code vectors-distributed-design.md}.
 *
 * <p>Tagged {@code @Tag("integration")} — requires Docker. Run with {@code ./gradlew
 * :vectors-storage:integrationTest}.
 */
@Tag("integration")
@Testcontainers(disabledWithoutDocker = true)
class S3StorageBackendIT {

  private static final String BUCKET = "test-vectors-storage";

  @Container
  static final LocalStackContainer LOCALSTACK =
      new LocalStackContainer(DockerImageName.parse("localstack/localstack:3.8"))
          .withServices(Service.S3);

  private static S3Client s3Client;
  private static final RangeHeaderCapture rangeCapture = new RangeHeaderCapture();

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
            .overrideConfiguration(
                ClientOverrideConfiguration.builder().addExecutionInterceptor(rangeCapture).build())
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

  @Test
  void putAndGetRoundTrip() throws IOException {
    S3StorageBackend b = backend();
    String key = "obj-" + UUID.randomUUID();
    byte[] value = "hello s3".getBytes();
    b.put(key, value);
    assertThat(b.get(key)).isEqualTo(value);
  }

  @Test
  void listReturnsKeysUnderPrefix() throws IOException {
    S3StorageBackend b = backend();
    String prefix = "list-" + UUID.randomUUID() + "/";
    b.put(prefix + "a", new byte[] {1});
    b.put(prefix + "b", new byte[] {2});
    b.put("other-" + UUID.randomUUID(), new byte[] {3});

    List<String> keys = b.list(prefix);

    assertThat(keys).containsExactlyInAnyOrder(prefix + "a", prefix + "b");
  }

  @Test
  void deleteRemovesKey() throws IOException {
    S3StorageBackend b = backend();
    String key = "delete-" + UUID.randomUUID();
    b.put(key, new byte[] {42});

    b.delete(key);

    assertThat(b.get(key)).isNull();
  }

  @Test
  void getMissingKeyReturnsNull() throws IOException {
    assertThat(backend().get("absent-" + UUID.randomUUID())).isNull();
  }

  @Test
  void getRange_partialFetchMatchesFullSlice() throws IOException {
    S3StorageBackend b = backend();
    String key = "blob-" + UUID.randomUUID();
    byte[] full = new byte[4096];
    for (int i = 0; i < full.length; i++) full[i] = (byte) (i & 0xFF);
    b.put(key, full);

    byte[] slice = b.getRange(key, 1024, 512);

    assertThat(slice).hasSize(512).isEqualTo(Arrays.copyOfRange(full, 1024, 1536));
  }

  @Test
  void getRange_usesHttpRangeHeader() throws IOException {
    S3StorageBackend b = backend();
    String key = "ranged-" + UUID.randomUUID();
    byte[] full = new byte[2048];
    for (int i = 0; i < full.length; i++) full[i] = (byte) i;
    b.put(key, full);

    rangeCapture.lastRange.set(null);
    b.getRange(key, 100, 50);

    assertThat(rangeCapture.lastRange.get())
        .as("S3StorageBackend.getRange must emit an HTTP Range header")
        .isEqualTo("bytes=100-149");
  }

  @Test
  void getRange_pastEofThrows() throws IOException {
    S3StorageBackend b = backend();
    String key = "small-" + UUID.randomUUID();
    b.put(key, new byte[] {1, 2, 3, 4});

    assertThatThrownBy(() -> b.getRange(key, 2, 5)).isInstanceOf(IndexOutOfBoundsException.class);
  }

  @Test
  void getRange_missingKeyReturnsNull() throws IOException {
    assertThat(backend().getRange("ghost-" + UUID.randomUUID(), 0, 4)).isNull();
  }

  @Test
  void conditionalPut_succeedsWhenKeyAbsent() throws IOException {
    S3StorageBackend b = backend();
    String key = "cas-new-" + UUID.randomUUID();

    StorageBackend.ConditionalPutResult result = b.conditionalPut(key, new byte[] {1, 2}, null);

    assertThat(result.succeeded()).isTrue();
    assertThat(result.newEtag()).isNotBlank();
    assertThat(b.get(key)).isEqualTo(new byte[] {1, 2});
  }

  @Test
  void conditionalPut_failsWhenKeyExistsAndEtagIsNull() throws IOException {
    S3StorageBackend b = backend();
    String key = "cas-exists-" + UUID.randomUUID();
    b.put(key, new byte[] {1});

    StorageBackend.ConditionalPutResult result = b.conditionalPut(key, new byte[] {2}, null);

    assertThat(result.succeeded()).isFalse();
    assertThat(result.newEtag()).isNull();
    assertThat(b.get(key)).isEqualTo(new byte[] {1});
  }

  @Test
  void conditionalPut_failsOnStaleEtag() throws IOException {
    S3StorageBackend b = backend();
    String key = "cas-stale-" + UUID.randomUUID();
    b.put(key, new byte[] {1});

    StorageBackend.ConditionalPutResult result =
        b.conditionalPut(key, new byte[] {2}, "stale-etag");

    assertThat(result.succeeded()).isFalse();
    assertThat(result.newEtag()).isNull();
    assertThat(b.get(key)).isEqualTo(new byte[] {1});
  }

  @Test
  void conditionalPut_succeedsWithCurrentEtag() throws IOException {
    S3StorageBackend b = backend();
    String key = "cas-current-" + UUID.randomUUID();
    StorageBackend.ConditionalPutResult first = b.conditionalPut(key, new byte[] {1}, null);
    assertThat(first.succeeded()).isTrue();

    StorageBackend.ConditionalPutResult second =
        b.conditionalPut(key, new byte[] {2}, first.newEtag());

    assertThat(second.succeeded()).isTrue();
    assertThat(second.newEtag()).isNotBlank();
    assertThat(b.get(key)).isEqualTo(new byte[] {2});
  }

  @Test
  void conditionalPut_allowsOnlyOneWriterForSameEtag() throws IOException {
    S3StorageBackend b = backend();
    String key = "cas-race-" + UUID.randomUUID();
    StorageBackend.ConditionalPutResult first = b.conditionalPut(key, new byte[] {1}, null);
    assertThat(first.succeeded()).isTrue();

    StorageBackend.ConditionalPutResult winner =
        b.conditionalPut(key, new byte[] {2}, first.newEtag());
    StorageBackend.ConditionalPutResult loser =
        b.conditionalPut(key, new byte[] {3}, first.newEtag());

    assertThat(winner.succeeded()).isTrue();
    assertThat(loser.succeeded()).isFalse();
    assertThat(b.get(key)).isEqualTo(new byte[] {2});
  }

  // ─── ExecutionInterceptor: captures outbound HTTP Range header ─────────────

  private static final class RangeHeaderCapture implements ExecutionInterceptor {
    final AtomicReference<String> lastRange = new AtomicReference<>();

    @Override
    public void beforeTransmission(
        Context.BeforeTransmission context, ExecutionAttributes executionAttributes) {
      SdkHttpRequest req = context.httpRequest();
      req.firstMatchingHeader("Range").ifPresent(lastRange::set);
    }
  }
}
