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
package com.integrallis.vectors.bench;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.Document;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.hnsw.HnswIndex;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * JMH benchmark isolating the {@link VectorCollection} facade overhead vs raw {@link HnswIndex}
 * search (G3 gap closure).
 *
 * <p>The facade adds: snapshot acquire (volatile read), tombstone filter pass, document-id
 * resolution, and metadata hydration on top of a direct HNSW search call. This benchmark quantifies
 * that overhead so callers can make informed decisions about using the facade vs the raw index API.
 *
 * <p>Run:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=VectorDbBenchmark
 * }</pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class VectorDbBenchmark {

  /** Corpus size. */
  @Param({"10000", "100000"})
  int n;

  /** Embedding dimension. */
  @Param({"128", "768"})
  int dim;

  private VectorCollection facadeHnsw;
  private HnswIndex rawHnsw;
  private float[] query;
  private SearchRequest facadeReq;

  @Setup(Level.Trial)
  public void setUp() {
    Random rng = new Random(55L);
    float[][] corpus = new float[n][dim];
    for (float[] row : corpus) {
      for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    }
    query = corpus[rng.nextInt(n)].clone();
    facadeReq = SearchRequest.builder(query, 10).searchListSize(100).build();

    // Facade path: VectorCollection wraps the index with snapshot + metadata layer
    facadeHnsw =
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.HNSW)
            .build();
    for (int i = 0; i < n; i++) {
      facadeHnsw.add(Document.of("doc-" + i, corpus[i]));
    }
    facadeHnsw.commit();

    // Raw path: direct HnswIndex with no facade
    rawHnsw =
        HnswIndex.builder(corpus, SimilarityFunction.EUCLIDEAN)
            .maxConnections(16)
            .efConstruction(200)
            .seed(42L)
            .build();
  }

  @TearDown(Level.Trial)
  public void tearDown() {
    facadeHnsw.close();
  }

  /**
   * Facade search: goes through snapshot acquire → tombstone filter → HNSW adapter → metadata
   * hydration.
   */
  @Benchmark
  public SearchResult searchFacadeHnsw(Blackhole bh) {
    SearchResult result = facadeHnsw.search(facadeReq);
    bh.consume(result);
    return result;
  }

  /**
   * Raw index search: direct HNSW search with no facade, no metadata, no snapshot overhead. The
   * difference between this and {@link #searchFacadeHnsw} is the net facade cost.
   */
  @Benchmark
  public com.integrallis.vectors.hnsw.SearchResult searchRawHnsw(Blackhole bh) {
    com.integrallis.vectors.hnsw.SearchResult result = rawHnsw.search(query, 10, 100);
    bh.consume(result);
    return result;
  }
}
