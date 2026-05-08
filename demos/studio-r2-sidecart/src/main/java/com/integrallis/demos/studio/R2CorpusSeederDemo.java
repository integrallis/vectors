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
package com.integrallis.demos.studio;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.ingest.BatchPolicy;
import com.integrallis.vectors.ingest.BulkIngestor;
import com.integrallis.vectors.ingest.IngestDoc;
import com.integrallis.vectors.ingest.IngestResult;
import com.integrallis.vectors.ingest.cursor.R2KeyCursor;
import com.integrallis.vectors.ingest.embedders.LangChain4jEmbedder;
import com.integrallis.vectors.ingest.sinks.DistributedVectorSink;
import com.integrallis.vectors.ingest.sources.IterableSource;
import com.integrallis.vectors.ivf.ClusterSplitter;
import com.integrallis.vectors.ivf.IvfBuildParams;
import com.integrallis.vectors.ivf.TierPolicy;
import com.integrallis.vectors.storage.backend.S3StorageBackend;
import com.integrallis.vectors.storage.backend.StorageBackend;
import com.integrallis.vectors.studio.distributed.PrefixedStorageBackend;
import com.integrallis.vectors.studio.sidecart.ingest.SidecartWriterSink;
import com.integrallis.vectors.studio.sidecart.sources.D1SidecartWriter;
import com.integrallis.vectors.studio.sidecart.sources.H2SidecartWriter;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.stream.Stream;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

/**
 * Refactored seeder CLI that drives the demo corpus through {@link BulkIngestor}. R2 + D1
 * credentials are read from a {@code .env} file (lookup order documented on {@link DemoEnv});
 * vectors land in a Cloudflare R2 prefix via {@link DistributedVectorSink}, and text rows land in
 * H2 or D1 via {@link SidecartWriterSink}.
 *
 * <p>Run via {@code ./gradlew :demos:studio-r2-sidecart:runSeed --args="--help"}.
 */
@Command(
    name = "r2-corpus-seeder-demo",
    description = "Seed an R2 prefix and a sidecart (H2 or D1) with the demo 24-doc corpus.",
    mixinStandardHelpOptions = true)
public final class R2CorpusSeederDemo implements Callable<Integer> {

  enum SidecartKind {
    NONE,
    H2,
    D1
  }

  private static final int DIM = 384;
  private static final SimilarityFunction METRIC = SimilarityFunction.COSINE;

  // ─── R2 target ───────────────────────────────────────────────────────────

  @Option(names = "--bucket", description = "R2 bucket (default: env VECTORS_R2_BUCKET)")
  private String bucket;

  @Option(names = "--prefix", description = "R2 key prefix (default: demo/)")
  private String prefix = "demo/";

  @Option(names = "--endpoint", description = "S3 endpoint URL (default: derived from account id)")
  private String endpoint;

  @Option(names = "--region", description = "S3 region (default: auto)")
  private String region;

  @Option(names = "--access-key", description = "R2 access key (default: VECTORS_R2_ACCESS_KEY)")
  private String accessKey;

  @Option(names = "--secret-key", description = "R2 secret key (default: VECTORS_R2_SECRET_KEY)")
  private String secretKey;

  @Option(names = "--wal-dir", description = "WAL directory (default: build/seed-wal)")
  private Path walDir;

  @Option(names = "--collection-name", description = "Logical collection name (default: demo)")
  private String collectionName = "demo";

  @Option(names = "--reset", description = "Delete every object under <bucket>/<prefix> first")
  private boolean reset;

  @Option(names = "--clean-wal", description = "Recursively delete the WAL dir before seeding")
  private boolean cleanWal;

  // ─── sidecart target ─────────────────────────────────────────────────────

  @Option(names = "--sidecart", description = "Sidecart kind: NONE (default), H2, or D1")
  private SidecartKind sidecart = SidecartKind.NONE;

  @Option(names = "--sidecart-table", description = "Sidecart table (default: docs)")
  private String sidecartTable = "docs";

  @Option(names = "--sidecart-id-column", description = "ID column (default: doc_id)")
  private String sidecartIdColumn = "doc_id";

  @Option(names = "--sidecart-text-column", description = "Text column (default: content)")
  private String sidecartTextColumn = "content";

  @Option(names = "--sidecart-blob-column", description = "Blob column (optional)")
  private String sidecartBlobColumn;

  @Option(names = "--sidecart-mime-column", description = "Mime column (default: mime_type)")
  private String sidecartMimeColumn = "mime_type";

  @Option(
      names = "--sidecart-h2-url",
      description = "H2 JDBC URL (default: jdbc:h2:./build/sidecart;AUTO_SERVER=TRUE)")
  private String sidecartH2Url = "jdbc:h2:./build/sidecart;AUTO_SERVER=TRUE";

  @Option(names = "--sidecart-h2-user", description = "H2 user (default: sa)")
  private String sidecartH2User = "sa";

  @Option(names = "--sidecart-h2-password", description = "H2 password (default: empty)")
  private String sidecartH2Password = "";

  @Option(names = "--sidecart-d1-account", description = "Cloudflare account id")
  private String sidecartD1Account;

  @Option(names = "--sidecart-d1-database", description = "D1 database id")
  private String sidecartD1Database;

  @Option(names = "--sidecart-d1-token", description = "Cloudflare API token")
  private String sidecartD1Token;

  public static void main(String[] args) {
    System.exit(new CommandLine(new R2CorpusSeederDemo()).execute(args));
  }

  @Override
  public Integer call() throws Exception {
    DemoEnv env = DemoEnv.load();
    R2Settings r2 = resolveR2(env);
    Path resolvedWalDir = (walDir != null) ? walDir : Paths.get("build", "seed-wal");

    System.out.printf(
        "[seed] R2: bucket=%s prefix=%s endpoint=%s region=%s%n",
        r2.bucket(), prefix, r2.endpoint(), r2.region());
    System.out.printf("[seed] WAL: %s%n", resolvedWalDir.toAbsolutePath());
    System.out.printf("[seed] sidecart: %s%n", sidecart);

    StorageBackend root =
        S3StorageBackend.create(
            URI.create(r2.endpoint()), r2.bucket(), r2.region(), r2.accessKey(), r2.secretKey());
    StorageBackend t3 =
        (prefix == null || prefix.isEmpty()) ? root : new PrefixedStorageBackend(root, prefix);

    if (reset) wipeR2Prefix(t3);
    if (cleanWal) wipeWalDir(resolvedWalDir);
    Files.createDirectories(resolvedWalDir);

    List<Doc> corpus = Corpus.realistic();
    System.out.printf(
        "[seed] ingesting %d documents (all-MiniLM-L6-v2, %d-dim)…%n", corpus.size(), DIM);

    DistributedVectorSink vectorSink =
        DistributedVectorSink.bootstrapping(
            resolvedWalDir,
            t3,
            new IvfBuildParams(8, 30, 0f, false, 42L, 0).withPq(16, 256, -1f),
            new ClusterSplitter(10_000, 30, 42L),
            new TierPolicy(5, 2),
            METRIC);

    com.integrallis.vectors.ingest.SidecartSink sidecartSink = openSidecart(env);

    try (BulkIngestor ingestor =
        BulkIngestor.builder()
            .vectorSink(vectorSink)
            .sidecartSink(sidecartSink)
            .embedder(new LangChain4jEmbedder(new AllMiniLmL6V2EmbeddingModel()))
            .batchPolicy(new BatchPolicy(256, 8L * 1024L * 1024L, Duration.ofSeconds(1)))
            .cursor(new R2KeyCursor(t3))
            .build()) {
      List<IngestDoc> docs = corpus.stream().map(d -> IngestDoc.text(d.id(), d.text())).toList();
      IngestResult r = ingestor.ingest(IterableSource.of("demo-corpus", docs));
      System.out.printf(
          "[seed] ingested %d docs in %s (%.1f docs/s)%n",
          r.docsCommitted(),
          r.totalDuration(),
          r.docsCommitted() * 1000.0 / Math.max(1L, r.totalDuration().toMillis()));
    }

    printNextSteps(r2, resolvedWalDir);
    return 0;
  }

  // ─── helpers ─────────────────────────────────────────────────────────────

  private R2Settings resolveR2(DemoEnv env) {
    String b = firstNonBlank(bucket, env.get("VECTORS_R2_BUCKET"));
    String ak = firstNonBlank(accessKey, env.get("VECTORS_R2_ACCESS_KEY"));
    String sk = firstNonBlank(secretKey, env.get("VECTORS_R2_SECRET_KEY"));
    String ep = firstNonBlank(endpoint, env.r2Endpoint());
    String rg = firstNonBlank(region, env.r2Region());
    requireConfigured(b, "--bucket / VECTORS_R2_BUCKET");
    requireConfigured(ak, "--access-key / VECTORS_R2_ACCESS_KEY");
    requireConfigured(sk, "--secret-key / VECTORS_R2_SECRET_KEY");
    requireConfigured(ep, "--endpoint / VECTORS_R2_ACCOUNT_ID");
    return new R2Settings(b, ep, rg, ak, sk);
  }

  private com.integrallis.vectors.ingest.SidecartSink openSidecart(DemoEnv env) {
    if (sidecart == SidecartKind.NONE) {
      return new com.integrallis.vectors.ingest.sinks.NoopSidecartSink();
    }
    if (sidecart == SidecartKind.H2) {
      H2SidecartWriter w =
          new H2SidecartWriter(
              sidecartH2Url,
              sidecartH2User,
              sidecartH2Password,
              sidecartTable,
              sidecartIdColumn,
              sidecartTextColumn,
              sidecartBlobColumn,
              sidecartMimeColumn);
      w.ensureSchema();
      return SidecartWriterSink.forH2(w);
    }
    String account = firstNonBlank(sidecartD1Account, env.cfAccountId());
    String database = firstNonBlank(sidecartD1Database, env.get("VECTORS_D1_DATABASE_ID"));
    String token = firstNonBlank(sidecartD1Token, env.get("VECTORS_CF_API_TOKEN"));
    requireConfigured(account, "--sidecart-d1-account / VECTORS_CF_ACCOUNT_ID");
    requireConfigured(database, "--sidecart-d1-database / VECTORS_D1_DATABASE_ID");
    requireConfigured(token, "--sidecart-d1-token / VECTORS_CF_API_TOKEN");
    D1SidecartWriter w =
        new D1SidecartWriter(
            account,
            database,
            token,
            sidecartTable,
            sidecartIdColumn,
            sidecartTextColumn,
            sidecartBlobColumn,
            sidecartMimeColumn);
    w.ensureSchema();
    return SidecartWriterSink.forD1(w);
  }

  private void wipeR2Prefix(StorageBackend t3) throws java.io.IOException {
    List<String> keys = t3.list("");
    int deleted = 0;
    for (String k : keys) {
      t3.delete(k);
      deleted++;
    }
    System.out.printf("[seed] --reset: deleted %d objects under prefix%n", deleted);
  }

  private void wipeWalDir(Path dir) throws java.io.IOException {
    if (!Files.exists(dir)) return;
    try (Stream<Path> walk = Files.walk(dir)) {
      walk.sorted(Comparator.reverseOrder())
          .forEach(
              p -> {
                try {
                  Files.deleteIfExists(p);
                } catch (java.io.IOException e) {
                  throw new RuntimeException("failed to delete " + p, e);
                }
              });
    }
    System.out.printf("[seed] --clean-wal: cleared %s%n", dir.toAbsolutePath());
  }

  private void printNextSteps(R2Settings r2, Path resolvedWalDir) {
    String connection =
        "r2://"
            + r2.bucket()
            + "/"
            + (prefix == null ? "" : prefix)
            + "?wal="
            + resolvedWalDir.toAbsolutePath()
            + "&dim="
            + DIM
            + "&metric="
            + METRIC.name()
            + "&name="
            + collectionName
            + "&endpoint="
            + r2.endpoint()
            + "&region="
            + r2.region();
    System.out.println();
    System.out.println("[seed] done. Studio connection string:");
    System.out.println();
    System.out.printf("  %s%n%n", connection);
  }

  private static String firstNonBlank(String a, String b) {
    if (a != null && !a.isBlank()) return a;
    if (b != null && !b.isBlank()) return b;
    return null;
  }

  private static void requireConfigured(String value, String label) {
    if (value == null || value.isBlank()) {
      throw new IllegalStateException("Missing required value: " + label);
    }
  }

  private record R2Settings(
      String bucket, String endpoint, String region, String accessKey, String secretKey) {}
}
