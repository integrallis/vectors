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
package com.integrallis.vectors.db;

import static org.junit.jupiter.api.Assertions.*;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import com.integrallis.vectors.core.filter.Filters;
import java.nio.file.Files;
import java.util.*;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordingFile;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class FlatCollectionBatchTest {
  private static final int DIM = 65;

  @Test
  void eligiblePublicBatchUsesAtMostOneTaskPerProcessor() throws Exception {
    org.junit.jupiter.api.Assumptions.assumeTrue(VectorUtil.supportsCosineNormReuse(DIM));
    int workers = Runtime.getRuntime().availableProcessors();
    var requests = requests(workers * 4, false);
    try (var collection = collection()) {
      collection.searchBatch(requests); // initialize the executor and scorer before recording
      var file = Files.createTempFile("flat-batch-workers-", ".jfr");
      try (var recording = new Recording()) {
        recording.enable("jdk.VirtualThreadStart");
        recording.start();
        var actual = collection.searchBatch(requests);
        recording.stop();
        recording.dump(file);
        long tasks =
            RecordingFile.readAllEvents(file).stream()
                .filter(event -> event.getEventType().getName().equals("jdk.VirtualThreadStart"))
                .count();
        System.out.println("public flat batch workers=" + tasks + " queries=" + requests.size());
        assertTrue(tasks > 0, "JFR must observe worker tasks");
        assertTrue(tasks <= workers, "batch should share scans across queries: " + tasks);
        assertResults(collection, requests, actual);
      } finally {
        Files.deleteIfExists(file);
      }
    }
  }

  @Test
  void batchRetainsProjectionCutoffsTombstonesTiesAndRequestOrder() {
    try (var collection = collection()) {
      var requests = requests(Runtime.getRuntime().availableProcessors() * 4 + 1, true);
      requests.set(
          0,
          SearchRequest.builder(collection.get("v0").vector().clone(), 10)
              .minScore(.9f)
              .includeVector(false)
              .includeText(false)
              .includeMetadata(false)
              .build());
      assertResults(collection, requests, collection.searchBatch(requests));
      collection.delete("v0");
      collection.commit();
      var actual = collection.searchBatch(requests);
      assertResults(collection, requests, actual);
      assertThrows(UnsupportedOperationException.class, () -> actual.add(null));
      // Caller-owned query vectors remain unchanged by batching.
      var first = requests.getFirst().query().clone();
      collection.searchBatch(requests);
      assertArrayEquals(first, requests.getFirst().query());
    }
  }

  @Test
  void filtersMixedKAndInvalidQueriesKeepIndividualSearchSemantics() {
    try (var collection = collection()) {
      var requests =
          new ArrayList<>(requests(Runtime.getRuntime().availableProcessors() * 4, false));
      requests.set(
          0,
          SearchRequest.builder(requests.getFirst().query(), 3)
              .filter(Filters.eq("group", 1))
              .build());
      assertResults(collection, requests, collection.searchBatch(requests));
      requests.set(
          0,
          SearchRequest.builder(requests.getFirst().query(), 10)
              .filter(Filters.eq("group", 1))
              .build());
      assertResults(collection, requests, collection.searchBatch(requests));
      requests.set(0, SearchRequest.builder(new float[DIM + 1], 10).build());
      var failure = assertThrows(RuntimeException.class, () -> collection.searchBatch(requests));
      assertInstanceOf(IllegalArgumentException.class, failure.getCause());
    }
  }

  @Test
  void closedCollectionKeepsTheBatchFailureCause() {
    var collection = collection();
    collection.close();
    var failure =
        assertThrows(
            RuntimeException.class,
            () ->
                collection.searchBatch(
                    requests(Runtime.getRuntime().availableProcessors() * 4, false)));
    assertInstanceOf(IllegalStateException.class, failure.getCause());
  }

  private static VectorCollection collection() {
    var collection =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .autoCommitThreshold(Integer.MAX_VALUE)
            .build();
    var random = new SplittableRandom(127731);
    float[] first = null;
    for (int i = 0; i < 512; i++) {
      float[] row = new float[DIM];
      for (int d = 0; d < DIM; d++) row[d] = (float) random.nextDouble(-1, 1);
      if (i == 0) first = row.clone();
      if (i == 100) row = first.clone();
      collection.add(
          new Document(
              "v" + i, row, "text-" + i, Map.of("group", MetadataValue.of((long) (i % 3)))));
    }
    collection.commit();
    return collection;
  }

  private static List<SearchRequest> requests(int count, boolean projections) {
    var random = new SplittableRandom(67219);
    var requests = new ArrayList<SearchRequest>();
    for (int i = 0; i < count; i++) {
      float[] query = new float[DIM];
      for (int d = 0; d < DIM; d++) query[d] = (float) random.nextDouble(-1, 1);
      requests.add(
          SearchRequest.builder(query, 10)
              .includeVector(!projections || i % 2 == 0)
              .includeText(!projections || i % 3 == 0)
              .includeMetadata(!projections || i % 4 == 0)
              .minScore(projections && i % 5 == 0 ? .9f : -Float.MAX_VALUE)
              .build());
    }
    return requests;
  }

  private static void assertResults(
      VectorCollection collection, List<SearchRequest> requests, List<SearchResult> actual) {
    assertEquals(requests.size(), actual.size());
    for (int q = 0; q < requests.size(); q++) {
      var expected = collection.search(requests.get(q));
      var got = actual.get(q);
      assertEquals(expected.hits().size(), got.hits().size());
      assertTrue(got.searchTimeNanos() >= 0);
      for (int i = 0; i < expected.hits().size(); i++) {
        var a = expected.hits().get(i);
        var b = got.hits().get(i);
        assertEquals(a.id(), b.id());
        assertEquals(a.score(), b.score(), 1e-6f);
        assertEquals(a.document().text(), b.document().text());
        assertEquals(a.document().metadata(), b.document().metadata());
        assertArrayEquals(a.document().vector(), b.document().vector());
      }
    }
  }
}
