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

import com.azure.storage.blob.BlobContainerClient;
import com.azure.storage.blob.BlobContainerClientBuilder;
import com.integrallis.vectors.storage.testkit.StorageBackendContract;
import java.io.IOException;
import java.util.List;
import java.util.UUID;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;

/**
 * Runs the full {@link StorageBackendContract} against a real Azure Blob container, or against the
 * Azurite emulator, through {@link AzureBlobStorageBackend}.
 *
 * <p>Opt-in and CI-safe: it self-skips unless {@code VECTORS_LIVE_AZURE_CONNECTION_STRING} is set,
 * so it never fails on a machine without credentials or an emulator.
 *
 * <pre>
 * # Azurite (npx -p azurite azurite-blob --blobPort 10000):
 * VECTORS_LIVE_AZURE_CONNECTION_STRING='DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=...;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;' \
 * VECTORS_LIVE_AZURE_CONTAINER=vectors-contract ./gradlew :vectors-storage:test --tests '*AzureBlobLive*'
 *
 * # A real storage account: the connection string from the portal, same variables.
 * </pre>
 *
 * <p><strong>Container hygiene:</strong> Every key the contract writes is prefixed with {@code
 * storage-contract-it/<run-uuid>/} (override with {@code VECTORS_LIVE_AZURE_PREFIX}). After all
 * tests finish, an {@code @AfterAll} lists every key under that prefix and deletes them. This still
 * requires a <em>dedicated test container</em> — do not run against a container that holds
 * production data.
 *
 * <p>The container is created if it does not exist (a test convenience; the backend itself never
 * creates containers).
 */
@Tag("live-object-store")
class AzureBlobLiveStorageBackendIT implements StorageBackendContract {

  private static final String RUN_UUID = UUID.randomUUID().toString();
  private static final String RUN_PREFIX;

  static {
    // A configured prefix is a parent, never the thing that gets deleted: the run's own UUID is
    // always appended, so the cleanup below can only remove keys this run wrote. Pointing the
    // variable at a prefix that already holds objects is therefore safe.
    String configured = System.getenv("VECTORS_LIVE_AZURE_PREFIX");
    String parent =
        configured == null || configured.isBlank()
            ? "storage-contract-it/"
            : (configured.endsWith("/") ? configured : configured + "/");
    RUN_PREFIX = parent + RUN_UUID + "/";
  }

  private static volatile StorageBackend sharedBackend;

  @Override
  public String key(String suffix) {
    return RUN_PREFIX + suffix + "-" + UUID.randomUUID();
  }

  @AfterAll
  static void deleteRunPrefix() throws IOException {
    StorageBackend b = sharedBackend;
    if (b == null) {
      return;
    }
    List<String> keys = b.list(RUN_PREFIX);
    for (String key : keys) {
      b.delete(key);
    }
  }

  @Override
  public StorageBackend backend() {
    String connectionString = System.getenv("VECTORS_LIVE_AZURE_CONNECTION_STRING");
    Assumptions.assumeTrue(
        connectionString != null && !connectionString.isBlank(),
        "set VECTORS_LIVE_AZURE_CONNECTION_STRING (and VECTORS_LIVE_AZURE_CONTAINER) to run the"
            + " Azure Blob contract");
    StorageBackend existing = sharedBackend;
    if (existing != null) {
      return existing;
    }
    synchronized (AzureBlobLiveStorageBackendIT.class) {
      if (sharedBackend == null) {
        String containerName = System.getenv("VECTORS_LIVE_AZURE_CONTAINER");
        if (containerName == null || containerName.isBlank()) {
          containerName = "vectors-contract";
        }
        BlobContainerClient client =
            new BlobContainerClientBuilder()
                .connectionString(connectionString)
                .containerName(containerName)
                .httpClient(AzureBlobStorageBackend.jdkHttpClient())
                .buildClient();
        client.createIfNotExists();
        sharedBackend = new AzureBlobStorageBackend(client);
      }
      return sharedBackend;
    }
  }
}
