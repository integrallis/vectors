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

import com.azure.core.http.HttpClient;
import com.azure.core.http.jdk.httpclient.JdkHttpClientBuilder;
import com.azure.core.http.rest.Response;
import com.azure.core.util.BinaryData;
import com.azure.core.util.Context;
import com.azure.storage.blob.BlobClient;
import com.azure.storage.blob.BlobContainerClient;
import com.azure.storage.blob.BlobContainerClientBuilder;
import com.azure.storage.blob.models.BlobDownloadContentResponse;
import com.azure.storage.blob.models.BlobDownloadResponse;
import com.azure.storage.blob.models.BlobItem;
import com.azure.storage.blob.models.BlobRange;
import com.azure.storage.blob.models.BlobRequestConditions;
import com.azure.storage.blob.models.BlobStorageException;
import com.azure.storage.blob.models.BlockBlobItem;
import com.azure.storage.blob.models.ListBlobsOptions;
import com.azure.storage.blob.options.BlobParallelUploadOptions;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * {@link StorageBackend} backed by Azure Blob Storage through the Azure SDK for Java.
 *
 * <p>Azure Blob is not S3-compatible, so it gets its own backend rather than an endpoint override.
 * The semantics match {@link S3StorageBackend} exactly, including the compare-and-swap the manifest
 * commit relies on, which Azure expresses with the same HTTP preconditions:
 *
 * <ul>
 *   <li>{@code expectedEtag == null} → {@code If-None-Match: *} — write only if the blob is absent
 *       (Azure answers {@code 409 BlobAlreadyExists} when it is not).
 *   <li>{@code expectedEtag != null} → {@code If-Match: "<etag>"} — write only if the etag matches
 *       ({@code 412 Precondition Failed} otherwise, {@code 404} when the blob is gone).
 * </ul>
 *
 * <p>ETags are returned without their surrounding double-quotes, as the other backends do. Keys map
 * to blob names one to one, so the {@code gen-N/...} layout and prefix listing work unchanged.
 *
 * <p>Thread-safe: all operations delegate to the thread-safe {@link BlobContainerClient}.
 *
 * <p>The Azure SDK is an opt-in runtime dependency ({@code vectors-storage-azure}); the base {@code
 * vectors-storage} artifact does not pull it in.
 */
public final class AzureBlobStorageBackend implements StorageBackend {
  private final BlobContainerClient container;

  /**
   * Wraps an already-built container client; the caller chose the credential (connection string,
   * SAS, or an {@code azure-identity} credential) and owns any resources behind it.
   *
   * @param container client for a container that must already exist
   */
  public AzureBlobStorageBackend(BlobContainerClient container) {
    this.container = Objects.requireNonNull(container, "container");
  }

  /**
   * Factory from a storage connection string (account key or SAS), the form the Azure portal and
   * Azurite hand out.
   *
   * @param connectionString an Azure Storage connection string
   * @param containerName container name (must already exist)
   */
  public static AzureBlobStorageBackend create(String connectionString, String containerName) {
    BlobContainerClient client =
        new BlobContainerClientBuilder()
            .connectionString(connectionString)
            .containerName(containerName)
            .httpClient(jdkHttpClient())
            .buildClient();
    return new AzureBlobStorageBackend(client);
  }

  /** The transport {@link #create} uses: the JDK {@code HttpClient}, no Netty. */
  public static HttpClient jdkHttpClient() {
    return new JdkHttpClientBuilder().build();
  }

  @Override
  public void put(String key, byte[] value) throws IOException {
    try {
      container.getBlobClient(key).upload(BinaryData.fromBytes(value), true);
    } catch (BlobStorageException e) {
      throw new IOException("Azure put failed for key: " + key, e);
    }
  }

  /**
   * Uploads {@code file} as one block blob; the SDK splits large files into blocks and commits them
   * as a single blob, so the query tier can still read it with ranged GETs.
   */
  @Override
  public void putFile(String key, Path file) throws IOException {
    try {
      container.getBlobClient(key).uploadFromFile(file.toString(), true);
    } catch (com.azure.core.exception.AzureException e) {
      throw new IOException("Azure putFile failed for key: " + key, e);
    }
  }

  @Override
  public byte[] get(String key) throws IOException {
    try {
      return container.getBlobClient(key).downloadContent().toBytes();
    } catch (BlobStorageException e) {
      if (e.getStatusCode() == 404) {
        return null;
      }
      throw new IOException("Azure get failed for key: " + key, e);
    }
  }

  @Override
  public StoredValue getWithEtag(String key) throws IOException {
    try {
      BlobDownloadContentResponse response =
          container.getBlobClient(key).downloadContentWithResponse(null, null, null, Context.NONE);
      return new StoredValue(
          response.getValue().toBytes(), unquoted(response.getDeserializedHeaders().getETag()));
    } catch (BlobStorageException e) {
      if (e.getStatusCode() == 404) {
        return null;
      }
      throw new IOException("Azure getWithEtag failed for key: " + key, e);
    }
  }

  @Override
  public InputStream open(String key) throws IOException {
    try {
      BlobClient blob = container.getBlobClient(key);
      if (!blob.exists()) {
        return null;
      }
      return blob.openInputStream();
    } catch (BlobStorageException e) {
      if (e.getStatusCode() == 404) {
        return null;
      }
      throw new IOException("Azure open failed for key: " + key, e);
    }
  }

  @Override
  public byte[] getRange(String key, long offset, int length) throws IOException {
    if (offset < 0 || length < 0 || offset > Long.MAX_VALUE - length) {
      throw new IndexOutOfBoundsException(
          "getRange(" + key + ", offset=" + offset + ", length=" + length + ")");
    }
    BlobClient blob = container.getBlobClient(key);
    if (length == 0) {
      // No zero-byte range request: check existence and bounds through the blob's size instead.
      try {
        long size = blob.getProperties().getBlobSize();
        if (offset > size) {
          throw new IndexOutOfBoundsException(
              "getRange(" + key + ", offset=" + offset + ", length=0) size=" + size);
        }
      } catch (BlobStorageException e) {
        if (e.getStatusCode() == 404) {
          return null;
        }
        throw new IOException("Azure getProperties failed for key: " + key, e);
      }
      return new byte[0];
    }
    try {
      ByteArrayOutputStream out = new ByteArrayOutputStream(length);
      BlobDownloadResponse response =
          blob.downloadStreamWithResponse(
              out, new BlobRange(offset, (long) length), null, null, false, null, Context.NONE);
      byte[] data = out.toByteArray();
      if (data.length != length) {
        // Azure clips a range that runs past the end instead of failing it.
        throw new IndexOutOfBoundsException(
            "getRange("
                + key
                + ", offset="
                + offset
                + ", length="
                + length
                + ") returned "
                + data.length
                + " bytes (past EOF); status "
                + response.getStatusCode());
      }
      return data;
    } catch (BlobStorageException e) {
      if (e.getStatusCode() == 404) {
        return null;
      }
      if (e.getStatusCode() == 416) {
        throw new IndexOutOfBoundsException(
            "getRange(" + key + ", offset=" + offset + ", length=" + length + ") past EOF");
      }
      throw new IOException("Azure getRange failed for key: " + key, e);
    }
  }

  @Override
  public List<String> list(String prefix) throws IOException {
    try {
      List<String> keys = new ArrayList<>();
      ListBlobsOptions options = new ListBlobsOptions().setPrefix(prefix);
      for (BlobItem item : container.listBlobs(options, null)) {
        keys.add(item.getName());
      }
      return keys;
    } catch (BlobStorageException e) {
      throw new IOException("Azure list failed for prefix: " + prefix, e);
    }
  }

  @Override
  public void delete(String key) throws IOException {
    try {
      container.getBlobClient(key).deleteIfExists();
    } catch (BlobStorageException e) {
      throw new IOException("Azure delete failed for key: " + key, e);
    }
  }

  @Override
  public ConditionalPutResult conditionalPut(String key, byte[] value, String expectedEtag)
      throws IOException {
    BlobRequestConditions conditions = new BlobRequestConditions();
    if (expectedEtag == null) {
      conditions.setIfNoneMatch("*");
    } else {
      conditions.setIfMatch(quoted(expectedEtag));
    }
    try {
      Response<BlockBlobItem> response =
          container
              .getBlobClient(key)
              .uploadWithResponse(
                  new BlobParallelUploadOptions(BinaryData.fromBytes(value))
                      .setRequestConditions(conditions),
                  null,
                  Context.NONE);
      return new ConditionalPutResult(true, unquoted(response.getValue().getETag()));
    } catch (BlobStorageException e) {
      int status = e.getStatusCode();
      if (status == 412 || status == 409 || status == 404) {
        // 412: If-Match mismatch. 409 BlobAlreadyExists: If-None-Match: * on an existing blob.
        // 404: an If-Match against a blob that no longer exists.
        return new ConditionalPutResult(false, null);
      }
      throw new IOException("Azure conditionalPut failed for key: " + key, e);
    }
  }

  // ─── etag helpers ─────────────────────────────────────────────────────────

  /** Strips surrounding double-quotes from an etag if present: {@code "abc"} → {@code abc}. */
  static String unquoted(String etag) {
    if (etag == null) {
      return null;
    }
    return (etag.startsWith("\"") && etag.endsWith("\""))
        ? etag.substring(1, etag.length() - 1)
        : etag;
  }

  /** Wraps an etag in double-quotes for a precondition header: {@code abc} → {@code "abc"}. */
  static String quoted(String etag) {
    if (etag == null) {
      return null;
    }
    return etag.startsWith("\"") ? etag : "\"" + etag + "\"";
  }
}
