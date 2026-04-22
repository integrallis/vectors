package com.integrallis.vectors.bench;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.Document;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.VectorCollectionBuilder;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;
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

/**
 * JMH benchmark comparing in-memory vs mmap-backed IVF-PQ search.
 *
 * <p>Three benchmark methods isolate three distinct cost components:
 *
 * <ul>
 *   <li>{@link #searchHeap(Void)} — control: search an in-memory {@code VectorCollection} built
 *       with no {@code storagePath}. Measures pure ADC + optional rescore cost.
 *   <li>{@link #searchMmap(Void)} — steady-state mmap: the collection has already been opened from
 *       disk once in {@link #setUp()} (pages are faulted in by JMH warmup). Measures the zero-copy
 *       mmap search path after page-cache amortisation.
 *   <li>{@link #openAndSearchMmap(Void)} — cold-start: opens a fresh {@code VectorCollection} from
 *       the on-disk generation on every invocation, then runs a single search. Captures the decode
 *       cost (codebook + code materialisation via {@code IvfIndex.decode}) and the first-touch
 *       page-fault amortisation amplified across the benchmark's iteration window.
 * </ul>
 *
 * <p>Run:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=IvfPqMmapSearchBenchmark
 * }</pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class IvfPqMmapSearchBenchmark {

  /** Corpus size. Kept modest so persistent build (K-Means + PQ training) completes quickly. */
  @Param({"10000", "100000"})
  int n;

  /** Vector dimension. {@code 16} subspaces divide both cleanly. */
  @Param({"128", "768"})
  int dim;

  /** PQ subspaces (M). Fixed at 16 — the sweep axis here is storage mode, not quantiser width. */
  @Param({"16"})
  int pqSubspaces;

  private static final int K = 10;
  private static final SimilarityFunction METRIC = SimilarityFunction.EUCLIDEAN;

  private float[][] corpus;
  private float[] query;
  private int nlist;
  private int nprobe;

  private Path storageRoot;
  private VectorCollection heapCollection;
  private VectorCollection mmapCollection;

  @Setup(Level.Trial)
  public void setUp() throws IOException {
    Random rng = new Random(77L);
    corpus = new float[n][dim];
    for (float[] row : corpus) {
      for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    }
    query = corpus[rng.nextInt(n)].clone();

    nlist = (int) Math.max(4, Math.ceil(Math.sqrt(n)));
    nprobe = (int) Math.max(1, Math.ceil(Math.sqrt(nlist)));

    // Heap-only baseline.
    heapCollection = newBuilder(null).build();
    addAllAndCommit(heapCollection);

    // Persist once: build + add + commit to disk, then close.
    storageRoot = Files.createTempDirectory("ivfpq-mmap-bench-").toAbsolutePath();
    try (VectorCollection persistent = newBuilder(storageRoot).build()) {
      addAllAndCommit(persistent);
    }
    // Reopen from the on-disk generation — this decodes IvfIndex and mmap's vectors.bin.
    mmapCollection = newBuilder(storageRoot).build();
  }

  @TearDown(Level.Trial)
  public void tearDown() throws IOException {
    if (heapCollection != null) heapCollection.close();
    if (mmapCollection != null) mmapCollection.close();
    if (storageRoot != null && Files.exists(storageRoot)) {
      try (Stream<Path> walk = Files.walk(storageRoot)) {
        walk.sorted(Comparator.reverseOrder()).forEach(IvfPqMmapSearchBenchmark::deleteQuiet);
      }
    }
  }

  private VectorCollectionBuilder newBuilder(Path storage) {
    VectorCollectionBuilder b =
        VectorCollection.builder()
            .dimension(dim)
            .metric(METRIC)
            .indexType(IndexType.IVF_PQ)
            .ivfK(nlist)
            .ivfNprobe(nprobe)
            .ivfPqSubspaces(pqSubspaces);
    if (storage != null) b.storagePath(storage);
    return b;
  }

  private void addAllAndCommit(VectorCollection col) {
    List<Document> batch = new ArrayList<>(1000);
    for (int i = 0; i < corpus.length; i++) {
      batch.add(Document.of("doc-" + i, corpus[i]));
      if (batch.size() == 1000) {
        col.addAll(batch);
        batch.clear();
      }
    }
    if (!batch.isEmpty()) col.addAll(batch);
    col.commit();
  }

  private static void deleteQuiet(Path p) {
    try {
      Files.deleteIfExists(p);
    } catch (IOException ignored) {
      // best-effort cleanup; tmpdir is reclaimed at reboot anyway
    }
  }

  @Benchmark
  public SearchResult searchHeap() {
    return heapCollection.search(SearchRequest.builder(query, K).build());
  }

  @Benchmark
  public SearchResult searchMmap() {
    return mmapCollection.search(SearchRequest.builder(query, K).build());
  }

  @Benchmark
  public int openAndSearchMmap() {
    try (VectorCollection col = newBuilder(storageRoot).build()) {
      return col.search(SearchRequest.builder(query, K).build()).hits().size();
    }
  }
}
