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
package com.integrallis.vectors.server.dto;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Pins that the index-specific tuning knobs added for the ANN-Benchmarks adapter (Vamana R/L/alpha,
 * IVF nlist/nprobe, IVF-PQ subspaces/clusters) parse from JSON and produce a working searchable
 * collection through {@link CreateCollectionRequest#toCollection}.
 */
@Tag("unit")
class CreateCollectionRequestTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final int DIM = 16;

  private static CreateCollectionRequest parse(String json) throws Exception {
    return MAPPER.readValue(json, CreateCollectionRequest.class);
  }

  private static float[] randomUnit(Random rng) {
    float[] v = new float[DIM];
    double norm = 0;
    for (int i = 0; i < DIM; i++) {
      v[i] = (float) rng.nextGaussian();
      norm += (double) v[i] * v[i];
    }
    float inv = (float) (1.0 / Math.sqrt(norm));
    for (int i = 0; i < DIM; i++) {
      v[i] *= inv;
    }
    return v;
  }

  private static void addSearchAssert(VectorCollection col) {
    Random rng = new Random(7L);
    float[] target = randomUnit(rng);
    col.add(Document.of("target", target));
    for (int i = 0; i < 300; i++) {
      col.add(Document.of("doc-" + i, randomUnit(rng)));
    }
    col.commit();
    SearchResult r = col.search(SearchRequest.builder(target, 5).build());
    assertThat(r.hits()).extracting(SearchResult.Hit::id).contains("target");
    col.close();
  }

  @Test
  void vamanaParamsParseAndBuildSearchableCollection() throws Exception {
    CreateCollectionRequest req =
        parse(
            "{\"name\":\"v\",\"dimension\":"
                + DIM
                + ",\"metric\":\"DOT_PRODUCT\",\"indexType\":\"VAMANA\","
                + "\"vamanaMaxDegree\":32,\"vamanaSearchListSize\":64,\"vamanaAlpha\":1.2,"
                + "\"vamanaBuildThreads\":2}");
    assertThat(req.validate()).isNull();
    assertThat(req.vamanaMaxDegree()).isEqualTo(32);
    assertThat(req.vamanaAlpha()).isEqualTo(1.2);
    addSearchAssert(req.toCollection(null));
  }

  @Test
  void ivfFlatParamsParseAndBuildSearchableCollection() throws Exception {
    CreateCollectionRequest req =
        parse(
            "{\"name\":\"i\",\"dimension\":"
                + DIM
                + ",\"metric\":\"DOT_PRODUCT\",\"indexType\":\"IVF_FLAT\","
                + "\"ivfK\":16,\"ivfNprobe\":8,\"ivfMaxIter\":20}");
    assertThat(req.validate()).isNull();
    assertThat(req.ivfK()).isEqualTo(16);
    assertThat(req.ivfNprobe()).isEqualTo(8);
    addSearchAssert(req.toCollection(null));
  }

  @Test
  void ivfPqParamsParseAndBuildSearchableCollection() throws Exception {
    CreateCollectionRequest req =
        parse(
            "{\"name\":\"p\",\"dimension\":"
                + DIM
                + ",\"metric\":\"DOT_PRODUCT\",\"indexType\":\"IVF_PQ\","
                + "\"ivfK\":16,\"ivfNprobe\":8,\"ivfPqSubspaces\":8,\"ivfPqClusters\":16,"
                + "\"ivfRescoreFactor\":4}");
    assertThat(req.validate()).isNull();
    assertThat(req.ivfPqSubspaces()).isEqualTo(8);
    addSearchAssert(req.toCollection(null));
  }

  @Test
  void negativeKnobIsRejected() throws Exception {
    CreateCollectionRequest req =
        parse(
            "{\"name\":\"x\",\"dimension\":"
                + DIM
                + ",\"metric\":\"COSINE\",\"indexType\":\"VAMANA\",\"vamanaMaxDegree\":-1}");
    assertThat(req.validate()).isEqualTo("vamanaMaxDegree must be positive");
  }

  @Test
  void lowAlphaIsRejected() throws Exception {
    CreateCollectionRequest req =
        parse(
            "{\"name\":\"x\",\"dimension\":"
                + DIM
                + ",\"metric\":\"COSINE\",\"indexType\":\"VAMANA\",\"vamanaAlpha\":0.5}");
    assertThat(req.validate()).isEqualTo("vamanaAlpha must be >= 1.0");
  }

  @Test
  void hnswStillWorksUnchanged() throws Exception {
    CreateCollectionRequest req =
        parse(
            "{\"name\":\"h\",\"dimension\":"
                + DIM
                + ",\"metric\":\"DOT_PRODUCT\",\"indexType\":\"HNSW\",\"hnswM\":16,"
                + "\"hnswEfConstruction\":100}");
    assertThat(req.validate()).isNull();
    addSearchAssert(req.toCollection(null));
  }
}
