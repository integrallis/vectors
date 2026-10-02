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

  public static void main(String[] args) throws Exception {
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
