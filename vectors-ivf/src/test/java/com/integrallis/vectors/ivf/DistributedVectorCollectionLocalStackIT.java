package com.integrallis.vectors.ivf;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.storage.backend.S3StorageBackend;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
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
 * P22 gate test — {@link DistributedVectorCollection} backed by a real S3 bucket via LocalStack.
 *
 * <p>Verifies:
 *
 * <ul>
 *   <li>WAL replay restores committed vectors after a crash-recovery cycle.
 *   <li>T3 cluster snapshots are durably stored in the S3 bucket.
 *   <li>S3 conditional-put prevents double-commit (real S3 ETag semantics).
 * </ul>
 *
 * <p>Tagged {@code @Tag("integration")} — requires Docker. Run via {@code ./gradlew
 * :vectors-ivf:integrationTest}.
 *
 * <p>Uses LocalStack 3.8, which provides a native S3 implementation compatible with S3 conditional
 * request headers ({@code If-None-Match: *}, {@code If-Match}).
 */
@Tag("integration")
@Testcontainers
class DistributedVectorCollectionLocalStackIT {

  private static final String BUCKET = "test-vectors";
  private static final int DIM = 32;
  private static final SimilarityFunction METRIC = SimilarityFunction.EUCLIDEAN;

  @Container
  static final LocalStackContainer LOCALSTACK =
      new LocalStackContainer(DockerImageName.parse("localstack/localstack:3.8"))
          .withServices(Service.S3);

  private static S3Client s3Client;

  @BeforeAll
  static void createBucket() {
    s3Client =
        S3Client.builder()
            .endpointOverride(LOCALSTACK.getEndpointOverride(Service.S3))
            .region(Region.of(LOCALSTACK.getRegion()))
            .credentialsProvider(
                StaticCredentialsProvider.create(
                    AwsBasicCredentials.create(
                        LOCALSTACK.getAccessKey(), LOCALSTACK.getSecretKey())))
            .forcePathStyle(true)
            .build();
    s3Client.createBucket(CreateBucketRequest.builder().bucket(BUCKET).build());
  }

  private S3StorageBackend t3Backend;

  @BeforeEach
  void freshBackend() {
    // Each test gets the same bucket; unique keys prevent cross-test interference.
    t3Backend = new S3StorageBackend(s3Client, BUCKET);
  }

  @AfterEach
  void closeBackend() throws Exception {
    // Do not close the shared s3Client; only close the backend wrapper.
    // (S3StorageBackend.close() closes the client — we manage lifecycle manually here.)
  }

  // ─── helpers ─────────────────────────────────────────────────────────────

  private float[][] randomVecs(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  private String[] ids(int n) {
    String[] a = new String[n];
    for (int i = 0; i < n; i++) a[i] = "doc-" + i;
    return a;
  }

  private DistributedVectorCollection build(
      float[][] vecs, String[] docIds, Path walDir, StorageBackend t3) throws IOException {
    IvfBuildParams params = new IvfBuildParams(4, 30, 0f, false, 42L);
    ClusterSplitter splitter = new ClusterSplitter(10_000, 30, 42L);
    TierPolicy policy = new TierPolicy(5, 2);
    return DistributedVectorCollection.build(
        vecs, docIds, METRIC, params, splitter, policy, walDir, t3);
  }

  // ─── tests ───────────────────────────────────────────────────────────────

  /**
   * Crash-recovery: build → add needle → commit → close → reopen from WAL. The needle must be found
   * after replay even though the JVM "restarted" (new collection instance from WAL).
   */
  @Test
  void walReplay_restoresCommittedVectors(@TempDir Path walDir) throws IOException {
    float[][] initial = randomVecs(100, DIM, 1L);

    try (var col = build(initial, ids(100), walDir, t3Backend)) {
      float[] needle = new float[DIM];
      Arrays.fill(needle, 80f);
      col.add("needle", needle);
      col.commit();
    }

    try (var col =
        DistributedVectorCollection.open(walDir, METRIC, new TierPolicy(5, 2), t3Backend)) {
      float[] needle = new float[DIM];
      Arrays.fill(needle, 80f);
      List<IvfHit> hits = col.search(needle, 1, 4);
      assertThat(hits).isNotEmpty();
      assertThat(hits.get(0).id()).isEqualTo("needle");
    }
  }

  /**
   * T3 snapshots: after a commit cycle, each cluster must have a snapshot object in the S3 bucket.
   */
  @Test
  void t3Backend_storesClusterSnapshotsInS3(@TempDir Path walDir) throws IOException {
    float[][] vecs = randomVecs(80, DIM, 2L);
    try (var col = build(vecs, ids(80), walDir, t3Backend)) {
      col.commit();
    }
    List<String> keys = t3Backend.list("cluster-");
    assertThat(keys).isNotEmpty();
  }

  /**
   * S3 conditional-put (real ETag): after a cluster snapshot is written, a conditional put with
   * {@code expectedEtag=null} ("must not exist") must be rejected by S3, preventing a
   * double-commit.
   *
   * <p>This test exercises real S3 ETag semantics via LocalStack. It requires LocalStack 3.x with
   * native S3 support for {@code If-None-Match: *} on PutObject.
   */
  @Test
  void conditionalPut_preventsDoubleCommit(@TempDir Path walDir) throws IOException {
    float[][] vecs = randomVecs(50, DIM, 3L);
    try (var col = build(vecs, ids(50), walDir, t3Backend)) {
      col.commit();
    }

    // The cluster-0 key was written by the commit; a conditional put asserting it does not exist
    // must fail.
    String key = "cluster-0";
    byte[] originalBytes = t3Backend.get(key);
    assertThat(originalBytes).isNotNull();

    StorageBackend.ConditionalPutResult result =
        t3Backend.conditionalPut(key, new byte[] {1, 2, 3}, null);
    assertThat(result.succeeded()).isFalse();
    assertThat(t3Backend.get(key)).isEqualTo(originalBytes);
  }
}
