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
  void rootRedirectsToCollections() throws Exception {
    HttpResponse<String> res = get("/");
    assertThat(res.statusCode()).isEqualTo(301);
    assertThat(res.headers().firstValue("location")).hasValue("/collections");
  }

  @Test
  void collectionsPageListsCollection() throws Exception {
    HttpResponse<String> res = get("/collections");
    assertThat(res.statusCode()).isEqualTo(200);
    assertThat(res.body()).contains("docs");
    assertThat(res.body()).contains("Vectors Studio");
  }

  @Test
  void datasetCatalogListsBundledEntriesUnloaded() throws Exception {
    HttpResponse<String> res = get("/api/datasets/catalog");
    assertThat(res.statusCode()).isEqualTo(200);
    // The bundled sample-datasets.json ships 8 entries; none share a name with the "docs" test
    // collection, so every entry reports loaded:false. No network is touched to serve the catalog.
    assertThat(countOccurrences(res.body(), "\"id\"")).isEqualTo(8);
    assertThat(countOccurrences(res.body(), "\"loaded\":false")).isEqualTo(8);
    assertThat(res.body()).doesNotContain("\"loaded\":true");
    assertThat(res.body()).contains("dbpedia-ada002");
  }

  @Test
  void datasetsPageRenders() throws Exception {
    HttpResponse<String> res = get("/datasets");
    assertThat(res.statusCode()).isEqualTo(200);
    assertThat(res.body()).contains("Sample datasets");
    assertThat(res.body()).contains("DBpedia");
    assertThat(res.body()).contains("/static/datasets.js");
  }

  private static int countOccurrences(String haystack, String needle) {
    int count = 0;
    int idx = 0;
    while ((idx = haystack.indexOf(needle, idx)) >= 0) {
      count++;
      idx += needle.length();
    }
    return count;
  }

  @Test
  void collectionOverviewRenders() throws Exception {
    HttpResponse<String> res = get("/collections/docs");
    assertThat(res.statusCode()).isEqualTo(200);
    assertThat(res.body()).contains("docs");
    assertThat(res.body()).contains("Search");
    assertThat(res.body()).contains("Visualize");
    assertThat(res.body()).contains("/collections/docs/projector");
    // Search form is now inlined on the overview page.
    assertThat(res.body()).contains("hx-post=\"/collections/docs/search\"");
    assertThat(res.body()).contains("name=\"query\"");
  }

  @Test
  void searchPostReturnsUnsupportedNotice() throws Exception {
    HttpResponse<String> res =
        client.send(
            HttpRequest.newBuilder(
                    URI.create("http://localhost:" + handle.port() + "/collections/docs/search"))
                .timeout(Duration.ofSeconds(5))
                .header("content-type", "application/x-www-form-urlencoded")
                .POST(HttpRequest.BodyPublishers.ofString("query=text&k=3"))
                .build(),
            BodyHandlers.ofString());
    assertThat(res.statusCode()).isEqualTo(200);
    // Returns the partial fragment (no full layout chrome).
    assertThat(res.body()).doesNotContain("<html");
    // Text search needs an embedding model the embedded backend lacks → clear notice, not results.
    assertThat(res.body()).contains("embedding model");
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
    HttpResponse<String> res =
        postJson("/api/collections/docs/search", "{\"id\":\"doc-0\",\"k\":3}");
    assertThat(res.statusCode()).isEqualTo(200);
    assertThat(res.body()).contains("\"hits\"");
    assertThat(res.body()).doesNotContain("doc-0\"");
  }

  @Test
  void searchByUnknownIdReturns404() throws Exception {
    HttpResponse<String> res =
        postJson("/api/collections/docs/search", "{\"id\":\"nope\",\"k\":3}");
    assertThat(res.statusCode()).isEqualTo(404);
  }

  @Test
  void searchByQueryRejectedWithoutEmbedder() throws Exception {
    // Text search needs an embedding model the embedded backend lacks → rejected, not fabricated.
    HttpResponse<String> res =
        postJson("/api/collections/docs/search", "{\"query\":\"text 0\",\"k\":3}");
    assertThat(res.statusCode()).isEqualTo(400);
    assertThat(res.body()).contains("embedding model");
  }

  @Test
  void mmrSearchDiversifiesNeighbours() throws Exception {
    // doc-0 (v[0]=1) has identical duplicates at every index where N % DIM == 0. Plain kNN returns
    // only those redundant duplicates; MMR must promote diverse (orthogonal) docs.
    HttpResponse<String> plain =
        postJson("/api/collections/docs/search", "{\"id\":\"doc-0\",\"k\":5}");
    HttpResponse<String> mmr =
        postJson("/api/collections/docs/search", "{\"id\":\"doc-0\",\"k\":5,\"mmr\":0.1}");
    assertThat(plain.statusCode()).isEqualTo(200);
    assertThat(mmr.statusCode()).isEqualTo(200);
    // Plain: every hit is a duplicate of doc-0 (index % DIM == 0).
    assertThat(hitIndices(plain.body())).isNotEmpty().allMatch(i -> i % DIM == 0);
    // MMR: at least one diverse (non-duplicate) hit is promoted.
    assertThat(hitIndices(mmr.body())).anyMatch(i -> i % DIM != 0);
  }

  private static java.util.List<Integer> hitIndices(String body) {
    var m = java.util.regex.Pattern.compile("\"id\"\\s*:\\s*\"doc-(\\d+)\"").matcher(body);
    var out = new java.util.ArrayList<Integer>();
    while (m.find()) out.add(Integer.parseInt(m.group(1)));
    return out;
  }
}
