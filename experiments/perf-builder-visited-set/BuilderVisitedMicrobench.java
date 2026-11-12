import java.util.Arrays;
import java.util.BitSet;

public class BuilderVisitedMicrobench {
  public static void main(String[] args) {
    int[] sizes = {1000, 5000, 20000, 100000};
    int iters = 50000;
    int warmup = 5000;

    System.out.println("size,iters,bitSetNs,genNs,speedup");
    for (int n : sizes) {
      BitSet bs = new BitSet(n);
      int[] tag = new int[n];
      int gen = 1;

      // Warmup
      for (int i = 0; i < warmup; i++) {
        bs.clear();
        bs.set(i % n);
      }
      for (int i = 0; i < warmup; i++) {
        if (++gen == 0) {
          Arrays.fill(tag, 0);
          gen = 1;
        }
        tag[i % n] = gen;
      }

      // BitSet benchmark
      long t0 = System.nanoTime();
      for (int i = 0; i < iters; i++) {
        bs.clear();
        bs.set(i % n);
      }
      long bitSetNs = System.nanoTime() - t0;

      // Generation-stamp benchmark
      gen = 1;
      t0 = System.nanoTime();
      for (int i = 0; i < iters; i++) {
        if (++gen == 0) {
          Arrays.fill(tag, 0);
          gen = 1;
        }
        tag[i % n] = gen;
      }
      long genNs = System.nanoTime() - t0;

      double speedup = (double) bitSetNs / genNs;
      System.out.printf("%d,%d,%d,%d,%.2f%n", n, iters, bitSetNs, genNs, speedup);
    }
  }
}
