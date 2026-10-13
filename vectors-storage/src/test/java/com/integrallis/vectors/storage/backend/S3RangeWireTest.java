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

import static org.assertj.core.api.Assertions.*;

import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.net.URI;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;

/** Actual AWS SDK HTTP requests to a bounded local S3-wire fixture; no mocked backend. */
@Tag("unit")
class S3RangeWireTest {
  @Test
  void rangeAndEmptyBoundsUseActualSdkWireContract() throws Exception {
    HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    AtomicReference<String> lastRange = new AtomicReference<>();
    server.createContext(
        "/",
        exchange -> {
          try {
            if (exchange.getRequestURI().getPath().endsWith("/missing")) {
              exchange.sendResponseHeaders(404, -1);
              return;
            }
            if (exchange.getRequestMethod().equals("HEAD")) {
              exchange.getResponseHeaders().set("Content-Length", "4");
              exchange.sendResponseHeaders(200, -1);
              return;
            }
            String range = exchange.getRequestHeaders().getFirst("Range");
            lastRange.set(range);
            if (!"bytes=1-2".equals(range)) {
              exchange.sendResponseHeaders(416, -1);
              return;
            }
            exchange.getResponseHeaders().set("Content-Range", "bytes 1-2/4");
            exchange.sendResponseHeaders(206, 2);
            exchange.getResponseBody().write(new byte[] {2, 3});
          } finally {
            exchange.close();
          }
        });
    server.start();
    try (S3Client client =
        S3Client.builder()
            .endpointOverride(URI.create("http://127.0.0.1:" + server.getAddress().getPort()))
            .region(Region.US_EAST_1)
            .credentialsProvider(
                StaticCredentialsProvider.create(AwsBasicCredentials.create("test", "test")))
            .forcePathStyle(true)
            .build()) {
      S3StorageBackend b = new S3StorageBackend(client, "bucket");
      assertThat(b.getRange("value", 1, 2)).containsExactly(2, 3);
      assertThat(lastRange.get()).isEqualTo("bytes=1-2");
      assertThat(b.getRange("value", 4, 0)).isEmpty();
      assertThatThrownBy(() -> b.getRange("value", 5, 0))
          .isInstanceOf(IndexOutOfBoundsException.class);
      assertThatThrownBy(() -> b.getRange("value", Long.MAX_VALUE, 2))
          .isInstanceOf(IndexOutOfBoundsException.class);
      assertThat(lastRange.get()).isEqualTo("bytes=1-2");
      assertThat(b.getRange("missing", 0, 0)).isNull();
    } finally {
      server.stop(0);
    }
  }
}
