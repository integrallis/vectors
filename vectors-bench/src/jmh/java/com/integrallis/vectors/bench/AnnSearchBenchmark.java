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

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.util.List;
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
 * JMH Tier-3: end-to-end ANN search throughput and recall via the public {@link VectorCollection}
 * facade.
 *
 * <p>Covers all four {@link IndexType} values: {@code FLAT}, {@code HNSW}, {@code VAMANA}, and
 * {@code IVF_FLAT}. Each benchmark is a single {@link VectorCollection#search(SearchRequest)} call
 * over a pre-built, in-memory collection, exercising the full facade path: snapshot acquire →
 * adapter dispatch → tombstone filter → metadata merge → result list assembly.
 *
 * <p>Run:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=AnnSearchBenchmark
 * }</pre>
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class AnnSearchBenchmark {

  @Param({"2000"})
  public int n;

  @Param({"128"})
  public int dim;

  @Param({"10"})
  public int k;

  private float[] query;

  private VectorCollection flatCol;
  private VectorCollection hnswCol;
  private VectorCollection vamanaCol;
  private VectorCollection ivfCol;

  @Setup(Level.Trial)
  public void setUp() {
    Random rng = new Random(11L);
    float[][] corpus = new float[n][dim];
    for (float[] row : corpus) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    query = corpus[0].clone();

    flatCol = buildAndCommit(IndexType.FLAT, corpus);
    hnswCol = buildAndCommit(IndexType.HNSW, corpus);
    vamanaCol = buildAndCommit(IndexType.VAMANA, corpus);
    ivfCol =
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.IVF_FLAT)
            .ivfK((int) Math.max(4, Math.sqrt(n)))
            .ivfNprobe(Math.max(1, (int) Math.sqrt(n) / 4))
            .build();
    for (int i = 0; i < n; i++) ivfCol.add(Document.of("doc-" + i, corpus[i]));
    ivfCol.commit();
  }

  private VectorCollection buildAndCommit(IndexType type, float[][] corpus) {
    VectorCollection col =
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(type)
            .build();
    for (int i = 0; i < corpus.length; i++) col.add(Document.of("doc-" + i, corpus[i]));
    col.commit();
    return col;
  }

  @TearDown(Level.Trial)
  public void tearDown() {
    flatCol.close();
    hnswCol.close();
    vamanaCol.close();
    ivfCol.close();
  }

  // --- FLAT brute-force scan ---

  @Benchmark
  public void searchFlat(Blackhole bh) {
    List<SearchResult.Hit> hits = flatCol.search(SearchRequest.builder(query, k).build()).hits();
    bh.consume(hits);
  }

  // --- HNSW approximate search ---

  @Benchmark
  public void searchHnsw(Blackhole bh) {
    List<SearchResult.Hit> hits = hnswCol.search(SearchRequest.builder(query, k).build()).hits();
    bh.consume(hits);
  }

  // --- Vamana approximate search ---

  @Benchmark
  public void searchVamana(Blackhole bh) {
    List<SearchResult.Hit> hits = vamanaCol.search(SearchRequest.builder(query, k).build()).hits();
    bh.consume(hits);
  }

  // --- IVF_FLAT approximate search ---

  @Benchmark
  public void searchIvfFlat(Blackhole bh) {
    List<SearchResult.Hit> hits = ivfCol.search(SearchRequest.builder(query, k).build()).hits();
    bh.consume(hits);
  }
}
