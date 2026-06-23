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
package com.integrallis.vectors.db;

import com.integrallis.vectors.db.storage.FileFormat;
import com.integrallis.vectors.storage.backend.LocalFileStorageBackend;
import com.integrallis.vectors.storage.backend.S3StorageBackend;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.Closeable;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiConsumer;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Stream;

/**
 * {@link GenerationSubscriber} that ships each committed {@code gen-NNNN/} directory to a {@link
 * StorageBackend} so read-replicas can pull it and serve reads (P3.1). Shipping is <b>asynchronous
 * and ordered</b>: {@link #onGenerationCommitted} enqueues to a single background thread and
 * returns immediately, so it never blocks the producer's commit.
 *
 * <p><b>Key layout</b> under the configured prefix mirrors the on-disk layout: every file of the
 * generation directory is written as {@code <prefix>gen-NNNN/<file>}, and only after all of them
 * land is {@code <prefix>CURRENT} written (an 8-byte little-endian generation number, identical to
 * the local {@code CURRENT} format). A follower that reads {@code CURRENT} therefore only ever sees
 * a generation whose payload is fully present — the same publish-last discipline as the local
 * commit. Use {@link GenerationSync#pull} on the follower to download into a local root, then
 * {@link VectorCollection#refresh()} to serve it.
 *
 * <p>Construct directly from a {@link StorageBackend} (the caller owns its lifecycle) or via {@link
 * #toUri(String)} for a {@code file://} or {@code s3://} target (this subscriber then owns and
 * closes the backend it created).
 */
public final class GenerationShippingSubscriber implements GenerationSubscriber, AutoCloseable {

  private static final Logger LOG = Logger.getLogger(GenerationShippingSubscriber.class.getName());

  private final StorageBackend backend;
  private final String keyPrefix; // "" or "some/prefix/"
  private final boolean ownsBackend;
  private final ExecutorService executor;
  private final BiConsumer<Long, Throwable> failureSink;
  private final AtomicLong shipFailureCount = new AtomicLong();
  private final AtomicReference<Throwable> lastShipFailure = new AtomicReference<>();

  /**
   * Ships to {@code backend} under {@code keyPrefix}. The caller retains ownership of {@code
   * backend} (it is not closed by {@link #close()}).
   *
   * @param backend the destination store
   * @param keyPrefix key prefix (a trailing {@code /} is added if missing; {@code ""} for the root)
   */
  public GenerationShippingSubscriber(StorageBackend backend, String keyPrefix) {
    this(backend, keyPrefix, false, null);
  }

  /**
   * Ships to {@code backend} under {@code keyPrefix} with an observable failure callback. Producer
   * commits never block on shipping (the work is enqueued to a daemon thread), so without this
   * visibility hook a follower can silently miss a generation when the destination put fails. The
   * sink is invoked from the shipping thread with the generation number and the underlying cause;
   * exceptions from the sink are caught and recorded as another failure (the pool thread is never
   * crashed by a broken sink). {@link #shipFailureCount()} and {@link #lastShipFailure()} expose
   * the same data for metric scraping.
   *
   * @param backend the destination store
   * @param keyPrefix key prefix (a trailing {@code /} is added if missing; {@code ""} for the root)
   * @param failureSink callback {@code (generationNumber, throwable)}; {@code null} disables it
   */
  public GenerationShippingSubscriber(
      StorageBackend backend, String keyPrefix, BiConsumer<Long, Throwable> failureSink) {
    this(backend, keyPrefix, false, failureSink);
  }

  private GenerationShippingSubscriber(
      StorageBackend backend, String keyPrefix, boolean ownsBackend) {
    this(backend, keyPrefix, ownsBackend, null);
  }

  private GenerationShippingSubscriber(
      StorageBackend backend,
      String keyPrefix,
      boolean ownsBackend,
      BiConsumer<Long, Throwable> failureSink) {
    this.backend = backend;
    this.keyPrefix = normalizePrefix(keyPrefix);
    this.ownsBackend = ownsBackend;
    this.failureSink = failureSink;
    this.executor =
        Executors.newSingleThreadExecutor(
            r -> {
              Thread t = new Thread(r, "vectors-generation-shipper");
              t.setDaemon(true);
              return t;
            });
  }

  /**
   * Creates a shipper for a replication target URI. Supported schemes:
   *
   * <ul>
   *   <li>{@code file:///abs/dir} — ships to a {@link LocalFileStorageBackend} rooted at the
   *       directory (empty key prefix).
   *   <li>{@code s3://bucket/optional/prefix} — ships to the S3 bucket under {@code
   *       optional/prefix}; region resolves from the {@code aws.region} system property or {@code
   *       AWS_REGION} env var, credentials from the default AWS chain.
   * </ul>
   *
   * <p>The target is an opaque blob store (its bytes are framed with etags), so a follower always
   * materialises generations with {@link GenerationSync#pull} into a clean local root before {@link
   * VectorCollection#refresh()} — it does not open the shipped directory in place.
   *
   * <p>{@code http(s)://} is not supported — there is no HTTP {@link StorageBackend}; pass a custom
   * {@link StorageBackend} to the constructor instead.
   *
   * @throws IllegalArgumentException if the scheme is unsupported or S3 region is unset
   * @throws UncheckedIOException if a {@code file://} backend cannot be created
   */
  public static GenerationShippingSubscriber toUri(String uri) {
    URI parsed = URI.create(uri);
    String scheme = parsed.getScheme();
    if (scheme == null) {
      throw new IllegalArgumentException("replication URI has no scheme: " + uri);
    }
    switch (scheme) {
      case "file" -> {
        Path root = Path.of(parsed.getPath());
        try {
          return new GenerationShippingSubscriber(new LocalFileStorageBackend(root), "", true);
        } catch (IOException e) {
          throw new UncheckedIOException("cannot open file replication target " + root, e);
        }
      }
      case "s3" -> {
        String bucket = parsed.getHost();
        if (bucket == null || bucket.isBlank()) {
          throw new IllegalArgumentException("s3 replication URI has no bucket: " + uri);
        }
        String region = System.getProperty("aws.region", System.getenv("AWS_REGION"));
        if (region == null || region.isBlank()) {
          throw new IllegalArgumentException(
              "s3 replication requires a region via the 'aws.region' system property or AWS_REGION"
                  + " env var");
        }
        String prefix = parsed.getPath() == null ? "" : parsed.getPath();
        return new GenerationShippingSubscriber(
            S3StorageBackend.create(bucket, region), prefix, true);
      }
      default ->
          throw new IllegalArgumentException(
              "unsupported replication scheme '" + scheme + "' (supported: file, s3)");
    }
  }

  @Override
  public void onGenerationCommitted(GenerationCommitEvent event) {
    // Non-blocking: hand the (immutable, finalized) generation directory to the shipping thread.
    executor.execute(() -> shipQuietly(event));
  }

  private void shipQuietly(GenerationCommitEvent event) {
    try {
      ship(event);
    } catch (Exception e) {
      // A replica that misses this generation will catch up on the next one (CURRENT is
      // monotonic), so log and continue rather than crash the shipping thread. Surface the
      // failure on observable counters and via the optional sink so operators can detect
      // "follower fell behind" without scraping logs.
      shipFailureCount.incrementAndGet();
      lastShipFailure.set(e);
      LOG.log(
          Level.WARNING,
          e,
          () -> "failed to ship generation " + event.generationNumber() + " to " + describe());
      BiConsumer<Long, Throwable> sink = failureSink;
      if (sink != null) {
        try {
          sink.accept(event.generationNumber(), e);
        } catch (Throwable sinkFailure) {
          // A broken sink must not crash the shipping thread; record it so it's still observable.
          shipFailureCount.incrementAndGet();
          lastShipFailure.set(sinkFailure);
          LOG.log(Level.WARNING, sinkFailure, () -> "failureSink threw while reporting a failure");
        }
      }
    }
  }

  /**
   * Number of generations whose shipping (or whose failure-sink invocation) has thrown since this
   * subscriber was constructed. A non-zero value means at least one follower may be missing the
   * corresponding generation; the next successful ship of a later generation will republish {@code
   * CURRENT} and the follower will catch up.
   */
  public long shipFailureCount() {
    return shipFailureCount.get();
  }

  /** Most recent throwable from a failed ship, or {@code null} if none has failed. */
  public Throwable lastShipFailure() {
    return lastShipFailure.get();
  }

  private void ship(GenerationCommitEvent event) throws IOException {
    String genDirName = fileName(event.generationDir());
    List<Path> files = new ArrayList<>();
    try (Stream<Path> s = Files.list(event.generationDir())) {
      s.filter(Files::isRegularFile).forEach(files::add);
    }
    // Ship all payload files first ...
    for (Path file : files) {
      byte[] bytes = Files.readAllBytes(file);
      backend.put(keyPrefix + genDirName + "/" + fileName(file), bytes);
    }
    // ... then publish CURRENT last, so a follower never observes a partial generation.
    byte[] current =
        ByteBuffer.allocate(Long.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
            .putLong(event.generationNumber())
            .array();
    backend.put(keyPrefix + FileFormat.CURRENT_FILE, current);
  }

  private String describe() {
    return backend.getClass().getSimpleName() + "[" + keyPrefix + "]";
  }

  /** Last path element as a string; never null for a generation directory or its files. */
  private static String fileName(Path path) {
    Path name = path.getFileName();
    if (name == null) {
      throw new IllegalArgumentException("path has no file name: " + path);
    }
    return name.toString();
  }

  /**
   * Stops the shipping thread (waiting briefly for in-flight uploads) and, if this subscriber
   * created the backend via {@link #toUri}, closes it.
   */
  @Override
  public void close() {
    executor.shutdown();
    try {
      if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
        executor.shutdownNow();
      }
    } catch (InterruptedException e) {
      executor.shutdownNow();
      Thread.currentThread().interrupt();
    }
    if (ownsBackend && backend instanceof Closeable c) {
      try {
        c.close();
      } catch (IOException e) {
        LOG.log(Level.WARNING, e, () -> "failed to close replication backend " + describe());
      }
    }
  }

  private static String normalizePrefix(String prefix) {
    if (prefix == null || prefix.isEmpty() || prefix.equals("/")) {
      return "";
    }
    String p = prefix.startsWith("/") ? prefix.substring(1) : prefix;
    return p.endsWith("/") ? p : p + "/";
  }
}
