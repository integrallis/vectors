package com.integrallis.vectors.bench;

import com.integrallis.vectors.quantization.ArrayVectorDataset;
import com.integrallis.vectors.quantization.BinaryMode;
import com.integrallis.vectors.quantization.BinaryQuantizer;
import com.integrallis.vectors.quantization.RaBitQuantizer;
import com.integrallis.vectors.quantization.ScalarBits;
import com.integrallis.vectors.quantization.ScalarQuantizer;
import com.integrallis.vectors.quantization.VectorDataset;
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
import org.openjdk.jmh.infra.Blackhole;

/**
 * JMH benchmark for quantizer training wall time (G1 gap closure).
 *
 * <p>Measures the one-time cost of {@code train()} for each quantizer family across realistic
 * dimension × dataset-size combinations. Training is the cold-path cost paid once before the search
 * hot path; this benchmark validates it stays within acceptable bounds at production dataset sizes.
 *
 * <p>Run:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=QuantizerTrainBenchmark
 * }</pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 4, time = 1)
public class QuantizerTrainBenchmark {

  /** Embedding dimension. */
  @Param({"128", "768", "1536"})
  int dim;

  /** Training dataset size. */
  @Param({"10000", "100000"})
  int n;

  private VectorDataset dataset;

  @Setup(Level.Trial)
  public void setUp() {
    Random rng = new Random(42L);
    float[][] vectors = new float[n][dim];
    for (float[] row : vectors) {
      for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    }
    dataset = new ArrayVectorDataset(vectors);
  }

  /**
   * Trains an INT8 scalar quantizer. Cost: one pass over dataset for confidence-interval quantile
   * computation; samples up to 25,000 vectors.
   */
  @Benchmark
  public ScalarQuantizer trainSQ8(Blackhole bh) {
    ScalarQuantizer sq = ScalarQuantizer.train(dataset, ScalarBits.INT8);
    bh.consume(sq);
    return sq;
  }

  /**
   * Trains an INT4 scalar quantizer. Same single-pass quantile computation as INT8 but packs two
   * values per byte.
   */
  @Benchmark
  public ScalarQuantizer trainSQ4(Blackhole bh) {
    ScalarQuantizer sq = ScalarQuantizer.train(dataset, ScalarBits.INT4);
    bh.consume(sq);
    return sq;
  }

  /**
   * Trains a BBQ (Balanced Binary Quantization) quantizer. Cost: one pass over dataset to compute
   * dataset centroid.
   */
  @Benchmark
  public BinaryQuantizer trainBBQ(Blackhole bh) {
    BinaryQuantizer bq = BinaryQuantizer.train(dataset, BinaryMode.BBQ);
    bh.consume(bq);
    return bq;
  }

  /**
   * Trains a RaBitQ (SIGMOD 2024) quantizer. Cost: centroid computation + random orthogonal
   * rotation matrix generation for padded dimension (O(dim²) memory; O(dim² * n) for dense random).
   */
  @Benchmark
  public RaBitQuantizer trainRaBitQ(Blackhole bh) {
    RaBitQuantizer rq = RaBitQuantizer.train(dataset, 42L);
    bh.consume(rq);
    return rq;
  }
}
