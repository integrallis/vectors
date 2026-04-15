package com.integrallis.vectors.storage.backend;

import java.io.Closeable;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.core.exception.SdkException;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.ListObjectsV2Request;
import software.amazon.awssdk.services.s3.model.ListObjectsV2Response;
import software.amazon.awssdk.services.s3.model.NoSuchKeyException;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.model.PutObjectResponse;
import software.amazon.awssdk.services.s3.model.S3Exception;
import software.amazon.awssdk.services.s3.model.S3Object;

/**
 * {@link StorageBackend} backed by an S3-compatible object store using AWS SDK v2.
 *
 * <p>Supports real AWS S3 (via {@link #create(String, String)}) and LocalStack / other
 * S3-compatible stores (via {@link #create(URI, String, String, String, String)}).
 *
 * <p>Conditional-put is implemented using S3 conditional request headers:
 *
 * <ul>
 *   <li>{@code expectedEtag == null} → {@code If-None-Match: *} — write only if key is absent.
 *   <li>{@code expectedEtag != null} → {@code If-Match: "<etag>"} — write only if etag matches.
 * </ul>
 *
 * On a precondition failure (HTTP 412 Precondition Failed or 409 Conflict for newer S3 API), {@link
 * #conditionalPut} returns {@code succeeded=false} without throwing.
 *
 * <p>ETags are stored and returned without surrounding double-quotes (e.g., {@code d41d8cd…}),
 * consistent with the format used by {@link HeapStorageBackend} and {@link
 * LocalFileStorageBackend}.
 *
 * <p>Thread-safe: all operations delegate to a thread-safe {@link S3Client}.
 */
public final class S3StorageBackend implements StorageBackend, Closeable {

  private final S3Client s3;
  private final String bucket;

  /**
   * Creates an {@code S3StorageBackend} wrapping an already-constructed {@link S3Client}. The
   * caller owns the client lifecycle; {@link #close()} will close it.
   *
   * @param s3 pre-configured S3 client
   * @param bucket S3 bucket name (must already exist)
   */
  public S3StorageBackend(S3Client s3, String bucket) {
    this.s3 = s3;
    this.bucket = bucket;
  }

  /**
   * Factory for real AWS S3. Credentials are resolved from the standard provider chain (env vars,
   * ~/.aws/credentials, instance profile, etc.).
   *
   * @param bucket S3 bucket name (must already exist)
   * @param region AWS region string (e.g., {@code "us-east-1"})
   */
  public static S3StorageBackend create(String bucket, String region) {
    return new S3StorageBackend(S3Client.builder().region(Region.of(region)).build(), bucket);
  }

  /**
   * Factory for LocalStack or other S3-compatible stores with a custom endpoint.
   *
   * @param endpoint custom endpoint URI (e.g., {@code http://localhost:4566})
   * @param bucket bucket name (must already exist)
   * @param region region string (e.g., {@code "us-east-1"})
   * @param accessKey access key credential
   * @param secretKey secret key credential
   */
  public static S3StorageBackend create(
      URI endpoint, String bucket, String region, String accessKey, String secretKey) {
    S3Client s3 =
        S3Client.builder()
            .endpointOverride(endpoint)
            .region(Region.of(region))
            .credentialsProvider(
                StaticCredentialsProvider.create(AwsBasicCredentials.create(accessKey, secretKey)))
            .forcePathStyle(true)
            .build();
    return new S3StorageBackend(s3, bucket);
  }

  @Override
  public void put(String key, byte[] value) throws IOException {
    try {
      s3.putObject(
          PutObjectRequest.builder().bucket(bucket).key(key).build(), RequestBody.fromBytes(value));
    } catch (SdkException e) {
      throw new IOException("S3 put failed for key: " + key, e);
    }
  }

  @Override
  public byte[] get(String key) throws IOException {
    try {
      return s3.getObjectAsBytes(GetObjectRequest.builder().bucket(bucket).key(key).build())
          .asByteArray();
    } catch (NoSuchKeyException e) {
      return null;
    } catch (SdkException e) {
      throw new IOException("S3 get failed for key: " + key, e);
    }
  }

  @Override
  public List<String> list(String prefix) throws IOException {
    try {
      List<String> keys = new ArrayList<>();
      String continuationToken = null;
      do {
        ListObjectsV2Request.Builder builder =
            ListObjectsV2Request.builder().bucket(bucket).prefix(prefix);
        if (continuationToken != null) builder.continuationToken(continuationToken);
        ListObjectsV2Response response = s3.listObjectsV2(builder.build());
        for (S3Object obj : response.contents()) keys.add(obj.key());
        continuationToken = response.isTruncated() ? response.nextContinuationToken() : null;
      } while (continuationToken != null);
      return keys;
    } catch (SdkException e) {
      throw new IOException("S3 list failed for prefix: " + prefix, e);
    }
  }

  @Override
  public void delete(String key) throws IOException {
    try {
      s3.deleteObject(DeleteObjectRequest.builder().bucket(bucket).key(key).build());
    } catch (SdkException e) {
      throw new IOException("S3 delete failed for key: " + key, e);
    }
  }

  @Override
  public ConditionalPutResult conditionalPut(String key, byte[] value, String expectedEtag)
      throws IOException {
    try {
      PutObjectRequest.Builder builder = PutObjectRequest.builder().bucket(bucket).key(key);
      if (expectedEtag == null) {
        builder.ifNoneMatch("*");
      } else {
        // S3 If-Match header requires the ETag in double-quoted form per RFC 7232.
        builder.ifMatch(quoted(expectedEtag));
      }
      PutObjectResponse response = s3.putObject(builder.build(), RequestBody.fromBytes(value));
      return new ConditionalPutResult(true, unquoted(response.eTag()));
    } catch (S3Exception e) {
      int status = e.statusCode();
      if (status == 412 || status == 409) {
        // 412 Precondition Failed: If-Match / If-None-Match mismatch.
        // 409 Conflict: returned by newer S3 API when If-None-Match: * fails.
        return new ConditionalPutResult(false, null);
      }
      throw new IOException("S3 conditionalPut failed for key: " + key, e);
    } catch (SdkException e) {
      throw new IOException("S3 conditionalPut failed for key: " + key, e);
    }
  }

  @Override
  public void close() {
    s3.close();
  }

  // ─── etag helpers ─────────────────────────────────────────────────────────

  /** Strips surrounding double-quotes from an S3 ETag if present: {@code "abc"} → {@code abc}. */
  private static String unquoted(String etag) {
    if (etag == null) return null;
    return (etag.startsWith("\"") && etag.endsWith("\""))
        ? etag.substring(1, etag.length() - 1)
        : etag;
  }

  /** Wraps an ETag in double-quotes for use in If-Match: {@code abc} → {@code "abc"}. */
  private static String quoted(String etag) {
    if (etag == null) return null;
    return etag.startsWith("\"") ? etag : "\"" + etag + "\"";
  }
}
