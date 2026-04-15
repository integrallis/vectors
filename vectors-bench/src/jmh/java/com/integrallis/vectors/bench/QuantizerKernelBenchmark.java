package com.integrallis.vectors.bench;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.quantization.ArrayVectorDataset;
import com.integrallis.vectors.quantization.BinaryQuantizedVectors;
import com.integrallis.vectors.quantization.BinaryQuantizer;
import com.integrallis.vectors.quantization.PQVectors;
import com.integrallis.vectors.quantization.ProductQuantizer;
import com.integrallis.vectors.quantization.RaBitQuantizedVectors;
import com.integrallis.vectors.quantization.RaBitQuantizer;
import com.integrallis.vectors.quantization.ScalarBits;
import com.integrallis.vectors.quantization.ScalarQuantizedVectors;
import com.integrallis.vectors.quantization.ScalarQuantizer;
import com.integrallis.vectors.quantization.ScoreFunction;
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
import org.openjdk.jmh.annotations.Warmup;

/**
 * JMH Tier-1 microbenchmarks for quantizer encode, decode, and score kernels.
 *
 * <p>Covers SQ8, PQ (ADC scoring), BQ, and RaBitQ at two typical dimensions. The {@code @Setup}
 * trains each quantizer offline so the timed loops measure only the per-query path.
 *
 * <p>Run via:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=QuantizerKernelBenchmark
 * }</pre>
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class QuantizerKernelBenchmark {

  /** Dimension must be divisible by PQ subspace count (dim/8). */
  @Param({"128", "768"})
  public int dim;

  private static final int CORPUS_SIZE = 10_000;

  // Query / encode inputs
  private float[] query;
  private byte[] sq8Encoded;
  private byte[] pqEncoded;
  private byte[] bqEncoded;
  private byte[] raqEncoded;

  // Quantizers
  private ScalarQuantizer sq8;
  private ProductQuantizer pq;
  private BinaryQuantizer bq;
  private RaBitQuantizer raq;

  // Pre-encoded corpora (for score-function benchmarks)
  private ScalarQuantizedVectors sq8Vectors;
  private PQVectors pqVectors;
  private BinaryQuantizedVectors bqVectors;
  private RaBitQuantizedVectors raqVectors;

  @Setup(Level.Trial)
  public void setUp() {
    Random rng = new Random(7L);
    float[][] data = new float[CORPUS_SIZE][dim];
    for (float[] row : data) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;

    query = data[0].clone();
    ArrayVectorDataset dataset = new ArrayVectorDataset(data);

    // SQ8
    sq8 = ScalarQuantizer.train(dataset, ScalarBits.INT8);
    sq8Encoded = sq8.encode(query);
    sq8Vectors = sq8.encodeAll(dataset);

    // PQ (dim/8 subspaces, 256 clusters, centered)
    int subspaces = Math.max(1, dim / 8);
    pq = ProductQuantizer.train(dataset, subspaces, 256, true);
    pqEncoded = pq.encode(query);
    pqVectors = pq.encodeAll(dataset);

    // BQ (BBQ mode)
    bq = BinaryQuantizer.train(dataset);
    bqEncoded = bq.encode(query);
    bqVectors = bq.encodeAll(dataset);

    // RaBitQ
    raq = RaBitQuantizer.train(dataset, 42L);
    raqEncoded = raq.encode(query);
    raqVectors = raq.encodeAll(dataset);
  }

  // --- SQ8 ---

  @Benchmark
  public byte[] sq8Encode() {
    return sq8.encode(query);
  }

  @Benchmark
  public float[] sq8Decode() {
    return sq8.decode(sq8Encoded);
  }

  @Benchmark
  public float sq8ScoreOrdinal0() {
    ScoreFunction fn = sq8Vectors.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
    return fn.score(0);
  }

  // --- PQ ---

  @Benchmark
  public byte[] pqEncode() {
    return pq.encode(query);
  }

  @Benchmark
  public float[] pqDecode() {
    return pq.decode(pqEncoded);
  }

  @Benchmark
  public float pqScoreOrdinal0() {
    ScoreFunction fn = pqVectors.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
    return fn.score(0);
  }

  // --- BQ ---

  @Benchmark
  public byte[] bqEncode() {
    return bq.encode(query);
  }

  @Benchmark
  public float[] bqDecode() {
    return bq.decode(bqEncoded);
  }

  @Benchmark
  public float bqScoreOrdinal0() {
    ScoreFunction fn = bqVectors.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
    return fn.score(0);
  }

  // --- RaBitQ ---

  @Benchmark
  public byte[] raqEncode() {
    return raq.encode(query);
  }

  @Benchmark
  public float[] raqDecode() {
    return raq.decode(raqEncoded);
  }

  @Benchmark
  public float raqScoreOrdinal0() {
    ScoreFunction fn = raqVectors.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
    return fn.score(0);
  }
}
