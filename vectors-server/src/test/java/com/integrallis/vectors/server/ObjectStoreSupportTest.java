/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
package com.integrallis.vectors.server;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.server.ServerConfig.ObjectStore;
import com.integrallis.vectors.server.ServerConfig.Provider;
import com.integrallis.vectors.storage.backend.AzureBlobStorageBackend;
import com.integrallis.vectors.storage.backend.S3StorageBackend;
import org.junit.jupiter.api.Test;

/** Provider dispatch of the object-storage durable floor. Clients are built, never contacted. */
class ObjectStoreSupportTest {

  @Test
  void sixArgumentConstructorIsS3() {
    ObjectStore cfg =
        new ObjectStore("http://127.0.0.1:9000", "bucket", "us-east-1", "key", "secret", null);
    assertThat(cfg.provider()).isEqualTo(Provider.S3);
    assertThat(cfg.prefix()).isEmpty();
    assertThat(ObjectStoreSupport.open(cfg)).isInstanceOf(S3StorageBackend.class);
  }

  @Test
  void s3WithoutEndpointStillOpensS3() {
    ObjectStore cfg = new ObjectStore("bucket", "bucket", "us-east-1", null, null, "p");
    ObjectStore aws = new ObjectStore(Provider.S3, null, "bucket", "us-east-1", null, null, "p");
    assertThat(cfg.provider()).isEqualTo(aws.provider());
    assertThat(ObjectStoreSupport.open(aws)).isInstanceOf(S3StorageBackend.class);
  }

  @Test
  void azureOpensBlobBackend() {
    ObjectStore cfg =
        new ObjectStore(Provider.AZURE, null, "container", null, "acct", "a2V5", "tenant/");
    assertThat(cfg.region()).isEmpty();
    assertThat(ObjectStoreSupport.open(cfg)).isInstanceOf(AzureBlobStorageBackend.class);
    assertThat(ObjectStoreSupport.keyPrefixFor(cfg, "docs")).isEqualTo("tenant/docs/");
  }

  @Test
  void azureConnectionStringDefaultsToPublicBlobEndpoint() {
    ObjectStore cfg = new ObjectStore(Provider.AZURE, null, "c", null, "acct", "a2V5", "");
    assertThat(ObjectStoreSupport.azureConnectionString(cfg))
        .isEqualTo(
            "DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=a2V5;"
                + "BlobEndpoint=https://acct.blob.core.windows.net;");
  }

  @Test
  void azureConnectionStringHonoursEndpointOverrideAndItsScheme() {
    ObjectStore cfg =
        new ObjectStore(
            Provider.AZURE,
            "http://127.0.0.1:10000/devstoreaccount1",
            "c",
            "ignored",
            "devstoreaccount1",
            "a2V5",
            "");
    assertThat(ObjectStoreSupport.azureConnectionString(cfg))
        .isEqualTo(
            "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=a2V5;"
                + "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;");
  }

  @Test
  void azureRequiresAccountNameAndKey() {
    assertThatThrownBy(() -> new ObjectStore(Provider.AZURE, null, "c", null, null, "k", ""))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("storage account name");
    assertThatThrownBy(() -> new ObjectStore(Provider.AZURE, null, "c", null, "acct", null, ""))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("storage account key");
  }

  @Test
  void s3StillRequiresRegion() {
    assertThatThrownBy(() -> new ObjectStore(Provider.S3, null, "b", null, null, null, ""))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("region");
  }

  @Test
  void providerParsing() {
    assertThat(Provider.parse(null)).isEqualTo(Provider.S3);
    assertThat(Provider.parse("  ")).isEqualTo(Provider.S3);
    assertThat(Provider.parse("S3")).isEqualTo(Provider.S3);
    assertThat(Provider.parse("Azure")).isEqualTo(Provider.AZURE);
    assertThat(Provider.parse("az")).isEqualTo(Provider.AZURE);
    assertThatThrownBy(() -> Provider.parse("gcs"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("gcs");
  }
}
