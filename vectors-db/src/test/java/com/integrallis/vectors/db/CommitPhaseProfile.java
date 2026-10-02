package com.integrallis.vectors.db;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.SimilarityFunction;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * TEMPORARY: where a commit's time goes at a realistic corpus size, per cadence.
 *
 * <p>Arguments: count dimension withText cadence[,cadence...]
 */
public final class CommitPhaseProfile {

  /** Sweeps efConstruction at a single commit, reporting build time against recall@10. */
  static void efSweep(int count, int dim, int queries, int[] efs) throws Exception {
    SplittableRandom data = new SplittableRandom(17);
    // Clustered, not uniform: uniform vectors in 512 dimensions are nearly equidistant, so recall@10
    // is near its worst case for every parameter and the sweep cannot rank anything.
    int clusters = 200;
    float[][] centres = new float[clusters][];
    for (int c = 0; c < clusters; c++) {
      centres[c] = unit(dim, data);
    }
    float[][] vectors = new float[count][];
    for (int i = 0; i < count; i++) {
      vectors[i] = jitter(centres[i % clusters], dim, 0.35, data);
    }
    float[][] qs = new float[queries][];
    for (int q = 0; q < queries; q++) {
      qs[q] = jitter(centres[q % clusters], dim, 0.35, data);
    }
    List<java.util.Set<String>> truth = new ArrayList<>();
    for (float[] q : qs) {
      truth.add(bruteForce(q, vectors, 10));
    }

    System.out.printf("%n%d clustered vectors x %d dims, %d queries, single commit%n", count, dim, queries);
    System.out.printf("%-6s %10s %10s %9s%n", "ef", "buildMs", "vec/s", "recall@10");
    for (int ef : efs) {
      Path dir = Files.createTempDirectory("ef" + ef);
      long start = System.nanoTime();
      double recall;
      try (VectorCollection c = VectorCollection.builder()
          .dimension(dim).metric(SimilarityFunction.COSINE).indexType(IndexType.HNSW)
          .hnswEfConstruction(ef)
          .storagePath(dir.toAbsolutePath()).build()) {
        List<Document> all = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
          all.add(Document.of("v" + i, vectors[i]));
        }
        c.addAll(all);
        c.commit();
        long ms = (System.nanoTime() - start) / 1_000_000;
        double hits = 0;
        for (int q = 0; q < qs.length; q++) {
          var got = new java.util.LinkedHashSet<String>();
          for (SearchResult.Hit h : c.search(SearchRequest.builder(qs[q], 10).build()).hits()) {
            got.add(h.id());
          }
          got.retainAll(truth.get(q));
          hits += got.size() / 10.0;
        }
        recall = hits / qs.length;
        System.out.printf("%-6d %10d %10.0f %9.4f%n", ef, ms, count * 1000.0 / ms, recall);
      }
      delete(dir);
    }
  }

  static float[] unit(int dim, SplittableRandom r) {
    float[] v = new float[dim];
    double s = 0;
    for (int d = 0; d < dim; d++) { v[d] = (float) r.nextGaussian(); s += v[d] * v[d]; }
    float n = (float) Math.sqrt(s);
    for (int d = 0; d < dim; d++) v[d] /= n;
    return v;
  }

  static float[] jitter(float[] centre, int dim, double spread, SplittableRandom r) {
    float[] v = new float[dim];
    double s = 0;
    for (int d = 0; d < dim; d++) {
      v[d] = (float) (centre[d] + r.nextGaussian() * spread);
      s += v[d] * v[d];
    }
    float n = (float) Math.sqrt(s);
    for (int d = 0; d < dim; d++) v[d] /= n;
    return v;
  }

  static java.util.Set<String> bruteForce(float[] q, float[][] vectors, int k) {
    Integer[] order = new Integer[vectors.length];
    for (int i = 0; i < vectors.length; i++) order[i] = i;
    java.util.Arrays.sort(order, (a, b) -> Float.compare(
        SimilarityFunction.COSINE.compare(q, vectors[b]), SimilarityFunction.COSINE.compare(q, vectors[a])));
    var out = new java.util.LinkedHashSet<String>();
    for (int i = 0; i < k; i++) out.add("v" + order[i]);
    return out;
  }

  public static void main(String[] args) throws Exception {
    if (args.length > 0 && args[0].equals("efsweep")) {
      efSweep(Integer.parseInt(args[1]), Integer.parseInt(args[2]), Integer.parseInt(args[3]),
          new int[] {64, 100, 128, 200, 320});
      return;
    }
    int count = args.length > 0 ? Integer.parseInt(args[0]) : 100_000;
    int dim = args.length > 1 ? Integer.parseInt(args[1]) : 512;
    boolean withText = args.length > 2 && Boolean.parseBoolean(args[2]);
    int[] cadences;
    if (args.length > 3) {
      String[] parts = args[3].split(",");
      cadences = new int[parts.length];
      for (int i = 0; i < parts.length; i++) {
        cadences[i] = Integer.parseInt(parts[i].trim());
      }
    } else {
      cadences = new int[] {count, count / 10};
    }

    System.out.printf("%d vectors x %d dims, text=%s, cores=%d%n",
        count, dim, withText, Runtime.getRuntime().availableProcessors());
    System.out.printf("%-9s %7s %9s | %7s %9s %8s %7s %8s | %8s %8s | %14s%n",
        "cadence", "gens", "totalMs", "materMs", "id+metaMs", "graphMs", "crcMs", "writeMs",
        "appendMs", "encodeMs", "bytes");

    for (int cadence : cadences) {
      Path dir = Files.createTempDirectory("phase");
      java.util.Arrays.fill(VectorCollectionImpl.PHASE, 0L);
      SplittableRandom random = new SplittableRandom(5);
      long gens;
      long start = System.nanoTime();
      try (VectorCollection c = VectorCollection.builder()
          .dimension(dim).metric(SimilarityFunction.COSINE).indexType(IndexType.HNSW)
          .storagePath(dir.toAbsolutePath()).build()) {
        List<Document> batch = new ArrayList<>(cadence);
        for (int i = 0; i < count; i++) {
          batch.add(doc(i, dim, random, withText));
          if (batch.size() == cadence) {
            c.addAll(batch);
            c.commit();
            batch.clear();
          }
        }
        if (!batch.isEmpty()) { c.addAll(batch); c.commit(); }
        gens = c.generationNumber();
      }
      long[] p = VectorCollectionImpl.PHASE;
      System.out.printf("%-9d %7d %9d | %7d %9d %8d %7d %8d | %8d %8d | %,14d%n",
          cadence, gens, (System.nanoTime() - start) / 1_000_000,
          p[0] / 1_000_000, p[1] / 1_000_000, p[2] / 1_000_000, p[3] / 1_000_000, p[4] / 1_000_000,
          p[5] / 1_000_000, p[6] / 1_000_000, occupied(dir));
      delete(dir);
    }
  }

  static Document doc(int i, int dim, SplittableRandom r, boolean withText) {
    float[] v = new float[dim];
    double s = 0;
    for (int d = 0; d < dim; d++) { v[d] = (float) r.nextGaussian(); s += v[d] * v[d]; }
    float n = (float) Math.sqrt(s);
    for (int d = 0; d < dim; d++) v[d] /= n;
    String text = withText
        ? "document " + i + " carrying roughly three hundred characters of stored text so that the "
            + "metadata payload is realistic for a corpus whose entries are encyclopedia abstracts, "
            + "which is the shape that makes metadata.bin the second largest payload after the "
            + "vectors themselves and therefore worth measuring separately from them here."
        : null;
    return new Document("v" + i, v, text,
        Map.of("category", MetadataValue.of("c" + (i % 14)), "n", MetadataValue.of(i)));
  }

  static long occupied(Path dir) throws Exception {
    java.util.Set<Object> seen = new java.util.HashSet<>();
    long total = 0;
    try (var walk = Files.walk(dir)) {
      for (Path p : (Iterable<Path>) walk.filter(Files::isRegularFile)::iterator) {
        Object key = Files.readAttributes(p, java.nio.file.attribute.BasicFileAttributes.class).fileKey();
        if (key != null && !seen.add(key)) continue;
        total += Files.size(p);
      }
    }
    return total;
  }

  static void delete(Path root) throws Exception {
    try (var w = Files.walk(root)) {
      w.sorted(Comparator.reverseOrder()).forEach(p -> { try { Files.delete(p); } catch (Exception ignored) {} });
    }
  }
}
