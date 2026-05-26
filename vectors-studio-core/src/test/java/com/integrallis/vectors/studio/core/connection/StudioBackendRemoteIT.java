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
package com.integrallis.vectors.studio.core.connection;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.server.ServerConfig;
import com.integrallis.vectors.server.VectorsServer;
import com.integrallis.vectors.server.client.DocumentPayload;
import com.integrallis.vectors.server.client.VectorsServerClient;
import com.integrallis.vectors.studio.core.search.DocumentView;
import com.integrallis.vectors.studio.core.search.SearchHit;
import com.integrallis.vectors.studio.core.search.SearchSpec;
import java.net.URI;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class StudioBackendRemoteIT {

  private static final int DIM = 4;
  private VectorsServer.ServerHandle handle;
  private RemoteStudioBackend backend;

  @BeforeEach
  void setUp() {
    handle = VectorsServer.start(ServerConfig.forTesting());
    try (VectorsServerClient client =
        new VectorsServerClient("http://localhost:" + handle.port())) {
      client.createCollection("docs", DIM, "COSINE", "FLAT", null);
      List<DocumentPayload> docs = new ArrayList<>();
      for (int i = 0; i < 20; i++) {
        float[] v = new float[DIM];
        v[i % DIM] = 1.0f;
        docs.add(new DocumentPayload("doc-" + i, v, "txt-" + i, null, null));
      }
      client.upsertDocuments("docs", docs);
    }
    backend =
        RemoteStudioBackend.open(
            new ConnectionConfig.Remote(
                URI.create("http://localhost:" + handle.port()), null, Duration.ofSeconds(5)));
  }

  @AfterEach
  void tearDown() {
    if (backend != null) backend.close();
    if (handle != null) handle.close();
  }

  @Test
  void listAndDescribeRoundTrip() {
    List<CollectionSummary> all = backend.listCollections();
    assertThat(all).extracting(CollectionSummary::name).contains("docs");
    CollectionSummary sum = backend.describe("docs");
    assertThat(sum.dimension()).isEqualTo(DIM);
    assertThat(sum.size()).isEqualTo(20);
  }

  @Test
  void previewDocumentsPaginates() {
    List<DocumentView> page = backend.previewDocuments("docs", 5, 3);
    assertThat(page).hasSize(3);
    assertThat(page.get(0).vector()).hasSize(DIM);
  }

  @Test
  void vectorBatchReturnsResolvedRows() {
    float[][] batch = backend.vectorBatch("docs", List.of("doc-0", "doc-1", "missing"));
    assertThat(batch).hasNumberOfRows(2);
  }

  @Test
  void searchReturnsHits() {
    float[] q = new float[DIM];
    q[0] = 1.0f;
    SearchSpec spec = new SearchSpec(q, null, 3, null, false, true, true);
    List<SearchHit> hits = backend.search("docs", spec);
    assertThat(hits).hasSize(3);
    assertThat(hits.get(0).score()).isGreaterThan(0.99);
    assertThat(hits.get(0).text()).startsWith("txt-");
  }

  @Test
  void remoteBackendForwardsBearerToken() {
    if (backend != null) {
      backend.close();
      backend = null;
    }
    if (handle != null) {
      handle.close();
      handle = null;
    }
    handle = VectorsServer.start(ServerConfig.forTesting().withApiKey("secret"));
    try (VectorsServerClient client =
        new VectorsServerClient("http://localhost:" + handle.port(), "secret")) {
      client.createCollection("docs", DIM, "COSINE", "FLAT", null);
    }
    backend =
        RemoteStudioBackend.open(
            new ConnectionConfig.Remote(
                URI.create("http://localhost:" + handle.port()), "secret", Duration.ofSeconds(5)));

    assertThat(backend.listCollections()).extracting(CollectionSummary::name).contains("docs");
  }
}
