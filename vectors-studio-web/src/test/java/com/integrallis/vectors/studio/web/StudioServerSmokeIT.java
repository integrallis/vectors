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
package com.integrallis.vectors.studio.web;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.studio.core.StudioSession;
import com.integrallis.vectors.studio.core.connection.EmbeddedStudioBackend;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.time.Duration;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class StudioServerSmokeIT {

  private static final int DIM = 4;
  private StudioServerHandle handle;
  private HttpClient client;

  @BeforeEach
  void setUp() {
    VectorCollection collection =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .build();
    for (int i = 0; i < 30; i++) {
      float[] v = new float[DIM];
      v[i % DIM] = 1.0f;
      collection.add(
          new Document("doc-" + i, v, "text " + i, Map.of("idx", MetadataValue.of((double) i))));
    }
    collection.commit();
    EmbeddedStudioBackend backend =
        EmbeddedStudioBackend.withCollections(Map.of("docs", collection));
    StudioSession session = new StudioSession(backend);
    handle = StudioServer.start(new StudioConfig(0, session));
    client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
  }

  @AfterEach
  void tearDown() {
    if (handle != null) handle.close();
  }

  private HttpResponse<String> get(String path) throws Exception {
    return client.send(
        HttpRequest.newBuilder(URI.create("http://localhost:" + handle.port() + path))
            .timeout(Duration.ofSeconds(5))
            .GET()
            .build(),
        BodyHandlers.ofString());
  }

  private HttpResponse<String> postJson(String path, String json) throws Exception {
    return client.send(
        HttpRequest.newBuilder(URI.create("http://localhost:" + handle.port() + path))
            .timeout(Duration.ofSeconds(5))
            .header("content-type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(json))
            .build(),
        BodyHandlers.ofString());
  }

  @Test
  void healthEndpointResponds() throws Exception {
    HttpResponse<String> res = get("/health");
    assertThat(res.statusCode()).isEqualTo(200);
    assertThat(res.body()).isEqualTo("OK");
  }

  @Test
  void homePageListsCollection() throws Exception {
    HttpResponse<String> res = get("/");
    assertThat(res.statusCode()).isEqualTo(200);
    assertThat(res.body()).contains("docs");
    assertThat(res.body()).contains("Vectors Studio");
  }

  @Test
  void collectionOverviewRenders() throws Exception {
    HttpResponse<String> res = get("/collections/docs");
    assertThat(res.statusCode()).isEqualTo(200);
    assertThat(res.body()).contains("docs");
    assertThat(res.body()).contains("Search");
    assertThat(res.body()).contains("Projector");
  }

  @Test
  void previewFragmentReturnsRows() throws Exception {
    HttpResponse<String> res = get("/collections/docs/preview?offset=0&limit=5");
    assertThat(res.statusCode()).isEqualTo(200);
    assertThat(res.body()).contains("doc-0");
    assertThat(res.body()).contains("doc-4");
  }

  @Test
  void unknownCollectionReturns404() throws Exception {
    assertThat(get("/collections/missing").statusCode()).isEqualTo(404);
  }

  @Test
  void recommenderEndpointReturnsHeuristic() throws Exception {
    HttpResponse<String> res = get("/collections/docs/recommend");
    assertThat(res.statusCode()).isEqualTo(200);
    assertThat(res.body()).containsAnyOf("PCA", "TSNE", "UMAP");
  }

  @Test
  void projectorPageLoads() throws Exception {
    HttpResponse<String> res = get("/collections/docs/projector");
    assertThat(res.statusCode()).isEqualTo(200);
    assertThat(res.body()).contains("/static/projector/app.js");
    assertThat(res.body()).contains("projector-shell");
    assertThat(res.body()).contains("data-tab=\"pca\"");
    assertThat(res.body()).contains("data-tab=\"tsne\"");
    assertThat(res.body()).contains("data-tab=\"umap\"");
    assertThat(res.body()).contains("id=\"tsne-perplexity\"");
    assertThat(res.body()).contains("id=\"umap-neighbors\"");
    assertThat(res.body()).contains("id=\"proj-pause\"");
    assertThat(res.body()).contains("id=\"ins-search\"");
    assertThat(res.body()).contains("id=\"ins-lasso\"");
  }

  @Test
  void metadataTsvIncludesMetadataColumns() throws Exception {
    HttpResponse<String> res = get("/api/collections/docs/metadata.tsv");
    assertThat(res.statusCode()).isEqualTo(200);
    String[] lines = res.body().split("\n");
    assertThat(lines[0]).isEqualTo("id\ttext\tidx");
    assertThat(lines).hasSizeGreaterThanOrEqualTo(2);
    assertThat(lines[1].split("\t")).hasSize(3);
  }

  @Test
  void searchByIdReturnsNeighbours() throws Exception {
    HttpResponse<String> res = postJson("/api/collections/docs/search", "{\"id\":\"doc-0\",\"k\":3}");
    assertThat(res.statusCode()).isEqualTo(200);
    assertThat(res.body()).contains("\"hits\"");
    assertThat(res.body()).doesNotContain("doc-0\"");
  }

  @Test
  void searchByUnknownIdReturns404() throws Exception {
    HttpResponse<String> res = postJson("/api/collections/docs/search", "{\"id\":\"nope\",\"k\":3}");
    assertThat(res.statusCode()).isEqualTo(404);
  }
}
