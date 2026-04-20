package com.integrallis.vectors.bench;

import com.integrallis.vectors.quantization.ArrayVectorDataset;
import com.integrallis.vectors.quantization.ProductQuantizer;
import com.integrallis.vectors.quantization.VectorDataset;
import java.util.Random;

/**
 * One-shot timing probe for PQ training: compares sequential and parallel paths on a synthetic 100k
 * × 768 corpus. Not part of the JMH suite; run standalone via the {@code pqTrainProbe} Gradle task.
 * Used only for pre-release performance validation of Round-2.E (Fix E).
 */
public final class PqTrainProbe {

  private PqTrainProbe() {}

  public static void main(String[] args) {
    int n = Integer.getInteger("probe.n", 100_000);
    int dim = Integer.getInteger("probe.dim", 768);
    int m = Integer.getInteger("probe.m", Math.max(1, dim / 8));
    int ks = Integer.getInteger("probe.ks", 256);
    int threads =
        Integer.getInteger(
            "probe.threads", Math.max(1, Runtime.getRuntime().availableProcessors() / 2));

    System.out.printf(
        "PQ train probe: n=%d dim=%d M=%d Ks=%d threads=%d%n", n, dim, m, ks, threads);

    // Generate synthetic gaussian data (deterministic seed for repeatability).
    System.out.print("Generating synthetic data... ");
    long gStart = System.nanoTime();
    float[][] data = new float[n][dim];
    Random rng = new Random(123L);
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        data[i][d] = (float) rng.nextGaussian();
      }
    }
    VectorDataset dataset = new ArrayVectorDataset(data);
    System.out.printf("%.1fs%n", (System.nanoTime() - gStart) / 1e9);

    // Warm up JIT + codebook caches with a small throwaway train.
    System.out.print("Warmup... ");
    long wStart = System.nanoTime();
    ProductQuantizer.train(dataset, Math.min(m, 4), ks, true, 1);
    System.out.printf("%.1fs%n", (System.nanoTime() - wStart) / 1e9);

    // Sequential baseline.
    System.out.print("Sequential (threads=1)... ");
    long seqStart = System.nanoTime();
    ProductQuantizer pqSeq = ProductQuantizer.train(dataset, m, ks, true, 1);
    double seqSec = (System.nanoTime() - seqStart) / 1e9;
    System.out.printf("%.2fs%n", seqSec);

    // Parallel run.
    System.out.printf("Parallel (threads=%d)... ", threads);
    long parStart = System.nanoTime();
    ProductQuantizer pqPar = ProductQuantizer.train(dataset, m, ks, true, threads);
    double parSec = (System.nanoTime() - parStart) / 1e9;
    System.out.printf("%.2fs%n", parSec);

    // Sanity: check encoded outputs are sensible (non-null, right shape).
    byte[] encSeq = pqSeq.encode(data[0]);
    byte[] encPar = pqPar.encode(data[0]);
    System.out.printf(
        "Codes: seq[0..4]=%d,%d,%d,%d par[0..4]=%d,%d,%d,%d%n",
        encSeq[0] & 0xFF,
        encSeq[1] & 0xFF,
        encSeq[2] & 0xFF,
        encSeq[3] & 0xFF,
        encPar[0] & 0xFF,
        encPar[1] & 0xFF,
        encPar[2] & 0xFF,
        encPar[3] & 0xFF);

    // MSE comparison on first 1000 vectors.
    double mseSeq = averageMse(pqSeq, data, 1000);
    double msePar = averageMse(pqPar, data, 1000);
    System.out.printf("MSE@1000: seq=%.6f par=%.6f%n", mseSeq, msePar);

    System.out.println();
    System.out.printf(
        "==> PQ train n=%d dim=%d M=%d Ks=%d: seq=%.2fs  par(T=%d)=%.2fs  speedup=%.2fx%n",
        n, dim, m, ks, seqSec, threads, parSec, seqSec / parSec);
  }

  private static double averageMse(ProductQuantizer pq, float[][] data, int limit) {
    int count = Math.min(limit, data.length);
    double total = 0.0;
    for (int i = 0; i < count; i++) {
      float[] orig = data[i];
      float[] recon = pq.decode(pq.encode(orig));
      double sum = 0.0;
      for (int d = 0; d < orig.length; d++) {
        double diff = (double) recon[d] - orig[d];
        sum += diff * diff;
      }
      total += sum / orig.length;
    }
    return total / count;
  }
}
