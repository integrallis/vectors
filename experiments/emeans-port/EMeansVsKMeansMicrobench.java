import com.integrallis.vectors.core.cluster.EMeans;
import com.integrallis.vectors.core.cluster.KMeans;
import java.util.Random;

/**
 * Quick manual timing comparison: e-means (5 epochs) vs Lloyd's k-means (50 iterations).
 *
 * <p>Run:
 * <pre>{@code
 *   javac -cp vectors-core/build/classes/java/main EMeansVsKMeansMicrobench.java
 *   java -cp .:vectors-core/build/classes/java/main EMeansVsKMeansMicrobench
 * }</pre>
 *
 * <p>Or from Gradle:
 * <pre>{@code
 *   ./gradlew :vectors-core:compileJava
 *   java -cp vectors-core/build/classes/java/main com.integrallis.vectors.core.cluster.EMeansVsKMeansMicrobench
 * }</pre>
 */
public class EMeansVsKMeansMicrobench {

  public static void main(String[] args) {
    int[] ns = {10_000, 50_000};
    int[] dims = {64, 128};
    int[] ks = {256, 1024};

    System.out.println("dataset,dim,k,algorithm,time_ms,error");
    Random rng = new Random(99L);

    for (int n : ns) {
      for (int dim : dims) {
        for (int k : ks) {
          float[][] dataset = randomDataset(n, dim, rng.nextLong());

          // Warmup both
          float[][] w1 = KMeans.train(dataset, k, 5, 42L);
          float[][] w2 = EMeans.train(dataset, k, 5.0f, 42L);

          // KMeans
          long t1 = System.nanoTime();
          float[][] ck = KMeans.train(dataset, k, 50, 42L);
          long t1e = System.nanoTime();
          double ek = quantizationError(dataset, ck);

          // EMeans
          long t2 = System.nanoTime();
          float[][] ce = EMeans.train(dataset, k, 5.0f, 42L);
          long t2e = System.nanoTime();
          double ee = quantizationError(dataset, ce);

          System.out.printf(
              "%d,%d,%d,KMeans-50,%.1f,%.6f%n",
              n, dim, k, (t1e - t1) / 1_000_000.0, ek);
          System.out.printf(
              "%d,%d,%d,EMeans-5,%.1f,%.6f%n",
              n, dim, k, (t2e - t2) / 1_000_000.0, ee);
        }
      }
    }
  }

  private static float[][] randomDataset(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) {
      for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    }
    return m;
  }

  private static double quantizationError(float[][] data, float[][] centroids) {
    double total = 0.0;
    for (float[] row : data) {
      float best = Float.MAX_VALUE;
      for (float[] c : centroids) {
        float dist = 0.0f;
        for (int d = 0; d < c.length; d++) {
          float diff = row[d] - c[d];
          dist += diff * diff;
        }
        if (dist < best) best = dist;
      }
      total += best;
    }
    return total / data.length;
  }
}
