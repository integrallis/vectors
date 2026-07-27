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

import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.core.ResponseInputStream;
import software.amazon.awssdk.core.exception.SdkException;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.AbortMultipartUploadRequest;
import software.amazon.awssdk.services.s3.model.CompleteMultipartUploadRequest;
import software.amazon.awssdk.services.s3.model.CompletedMultipartUpload;
import software.amazon.awssdk.services.s3.model.CompletedPart;
import software.amazon.awssdk.services.s3.model.CreateMultipartUploadRequest;
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;
import software.amazon.awssdk.services.s3.model.ListObjectsV2Request;
import software.amazon.awssdk.services.s3.model.ListObjectsV2Response;
import software.amazon.awssdk.services.s3.model.NoSuchKeyException;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.model.PutObjectResponse;
import software.amazon.awssdk.services.s3.model.S3Exception;
import software.amazon.awssdk.services.s3.model.S3Object;
import software.amazon.awssdk.services.s3.model.UploadPartRequest;
import software.amazon.awssdk.services.s3.model.UploadPartResponse;

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

  /**
   * Part size for multipart uploads. 256 MiB × 10000-part cap = 2.5 TB, ample for a 100M {@code
   * vectors.bin}.
   */
  private static final int PART_SIZE = 256 << 20;

  /**
   * Uploads {@code file} as a single S3 object, streaming it so no {@code byte[]} of the whole file
   * is ever allocated. Files at or below one part go via a single {@code putObject}; larger files
   * use a multipart upload (create → uploadPart per {@value #PART_SIZE}-byte slice → complete),
   * aborting the upload on any failure. This is the 100M-scale {@code vectors.bin} path — one
   * addressable object the query tier reads with ranged GETs.
   */
  @Override
  public void putFile(String key, Path file) throws IOException {
    long size = Files.size(file);
    if (size <= PART_SIZE) {
      try {
        s3.putObject(
            PutObjectRequest.builder().bucket(bucket).key(key).build(), RequestBody.fromFile(file));
      } catch (SdkException e) {
        throw new IOException("S3 putFile failed for key: " + key, e);
      }
      return;
    }
    String uploadId;
    try {
      uploadId =
          s3.createMultipartUpload(
                  CreateMultipartUploadRequest.builder().bucket(bucket).key(key).build())
              .uploadId();
    } catch (SdkException e) {
      throw new IOException("S3 createMultipartUpload failed for key: " + key, e);
    }
    List<CompletedPart> parts = new ArrayList<>();
    try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
      long pos = 0;
      int partNumber = 1;
      while (pos < size) {
        int len = (int) Math.min(PART_SIZE, size - pos);
        byte[] part = new byte[len];
        ByteBuffer bb = ByteBuffer.wrap(part);
        long p = pos;
        while (bb.hasRemaining()) {
          int r = ch.read(bb, p);
          if (r < 0) {
            throw new IOException("unexpected EOF reading " + file + " at " + p);
          }
          p += r;
        }
        UploadPartResponse resp =
            s3.uploadPart(
                UploadPartRequest.builder()
                    .bucket(bucket)
                    .key(key)
                    .uploadId(uploadId)
                    .partNumber(partNumber)
                    .build(),
                RequestBody.fromBytes(part));
        parts.add(CompletedPart.builder().partNumber(partNumber).eTag(resp.eTag()).build());
        pos += len;
        partNumber++;
      }
      s3.completeMultipartUpload(
          CompleteMultipartUploadRequest.builder()
              .bucket(bucket)
              .key(key)
              .uploadId(uploadId)
              .multipartUpload(CompletedMultipartUpload.builder().parts(parts).build())
              .build());
    } catch (IOException | RuntimeException e) {
      try {
        s3.abortMultipartUpload(
            AbortMultipartUploadRequest.builder()
                .bucket(bucket)
                .key(key)
                .uploadId(uploadId)
                .build());
      } catch (SdkException ignore) {
        // best-effort cleanup of the dangling upload
      }
      throw new IOException("S3 multipart putFile failed for key: " + key, e);
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
  public StoredValue getWithEtag(String key) throws IOException {
    try {
      var response =
          s3.getObjectAsBytes(GetObjectRequest.builder().bucket(bucket).key(key).build());
      return new StoredValue(response.asByteArray(), unquoted(response.response().eTag()));
    } catch (NoSuchKeyException e) {
      return null;
    } catch (SdkException e) {
      throw new IOException("S3 getWithEtag failed for key: " + key, e);
    }
  }

  @Override
  public InputStream open(String key) throws IOException {
    try {
      ResponseInputStream<GetObjectResponse> response =
          s3.getObject(GetObjectRequest.builder().bucket(bucket).key(key).build());
      return response;
    } catch (NoSuchKeyException e) {
      return null;
    } catch (SdkException e) {
      throw new IOException("S3 open failed for key: " + key, e);
    }
  }

  @Override
  public byte[] getRange(String key, long offset, int length) throws IOException {
    if (offset < 0 || length < 0) {
      throw new IndexOutOfBoundsException(
          "getRange(" + key + ", offset=" + offset + ", length=" + length + ")");
    }
    if (length == 0) {
      // Avoid issuing a zero-byte Range request (some S3 implementations reject it).
      // Still verify the key exists so missing keys return null per the contract.
      try {
        s3.headObject(b -> b.bucket(bucket).key(key));
      } catch (NoSuchKeyException e) {
        return null;
      } catch (SdkException e) {
        throw new IOException("S3 head failed for key: " + key, e);
      }
      return new byte[0];
    }
    long endInclusive = offset + length - 1L;
    String range = "bytes=" + offset + "-" + endInclusive;
    try {
      byte[] data =
          s3.getObjectAsBytes(
                  GetObjectRequest.builder().bucket(bucket).key(key).range(range).build())
              .asByteArray();
      if (data.length != length) {
        // S3 returned fewer bytes than requested → range extended past EOF.
        throw new IndexOutOfBoundsException(
            "getRange("
                + key
                + ", offset="
                + offset
                + ", length="
                + length
                + ") returned "
                + data.length
                + " bytes (past EOF)");
      }
      return data;
    } catch (NoSuchKeyException e) {
      return null;
    } catch (S3Exception e) {
      if (e.statusCode() == 416) {
        // 416 Range Not Satisfiable: offset is at or beyond the object size.
        throw new IndexOutOfBoundsException(
            "getRange(" + key + ", offset=" + offset + ", length=" + length + ") past EOF");
      }
      throw new IOException("S3 getRange failed for key: " + key, e);
    } catch (SdkException e) {
      throw new IOException("S3 getRange failed for key: " + key, e);
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
        // Rely solely on the If-Match precondition: S3 rejects a stale ETag server-side and the
        // catch below maps 412/409/404 to succeeded=false. This drops a redundant HEAD round-trip.
        // S3 If-Match header requires the ETag in double-quoted form per RFC 7232.
        builder.ifMatch(quoted(expectedEtag));
      }
      PutObjectResponse response = s3.putObject(builder.build(), RequestBody.fromBytes(value));
      return new ConditionalPutResult(true, unquoted(response.eTag()));
    } catch (S3Exception e) {
      int status = e.statusCode();
      if (status == 412 || status == 409 || status == 404) {
        // 412 Precondition Failed: If-Match / If-None-Match mismatch.
        // 409 Conflict: returned by newer S3 API when If-None-Match: * fails.
        // 404 Not Found: expected an existing ETag but the key is absent.
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
