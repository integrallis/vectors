/* Copyright 2026 Integrallis Software, LLC. Licensed under the Apache License, Version 2.0. */
import com.integrallis.vectors.core.*;
import com.integrallis.vectors.db.*;
import java.nio.file.*;

/** Build once with baseline jars; both benchmark arms reopen copies of the same generation. */
public class BuildPublicFixture {
  public static void main(String[] args) throws Exception {
    var rows = QueryPerformance.read(Path.of(args[0]));
    var path = Path.of(args[1]);
    if (Files.exists(path)) throw new IllegalArgumentException("Fixture already exists: " + path);
    try (var c = VectorCollection.builder().dimension(rows[0].length)
        .metric(SimilarityFunction.COSINE).indexType(IndexType.HNSW)
        .storagePath(path).autoCommitThreshold(Integer.MAX_VALUE).build()) {
      for (int i=0;i<20000;i++) c.add(Document.of("row-"+i,rows[i]));
      c.commit();
    }
    System.out.println("Built baseline GLOVE 20k public fixture: " + path);
  }
}
