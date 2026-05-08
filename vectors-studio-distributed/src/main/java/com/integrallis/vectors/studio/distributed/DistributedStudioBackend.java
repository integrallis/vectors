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
package com.integrallis.vectors.studio.distributed;

import com.integrallis.vectors.ivf.DistributedVectorCollection;
import com.integrallis.vectors.ivf.IvfHit;
import com.integrallis.vectors.storage.backend.S3StorageBackend;
import com.integrallis.vectors.storage.backend.StorageBackend;
import com.integrallis.vectors.studio.core.connection.CollectionSummary;
import com.integrallis.vectors.studio.core.connection.StudioBackend;
import com.integrallis.vectors.studio.core.search.DocumentPageView;
import com.integrallis.vectors.studio.core.search.DocumentView;
import com.integrallis.vectors.studio.core.search.SearchHit;
import com.integrallis.vectors.studio.core.search.SearchSpec;
import com.integrallis.vectors.studio.distributed.search.RrfFuser;
import com.integrallis.vectors.studio.sidecart.SidecartRecord;
import com.integrallis.vectors.studio.sidecart.SidecartSource;
import com.integrallis.vectors.studio.sidecart.TextSearchHit;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.IntConsumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adapts a single Cloudflare R2 / S3-backed {@link DistributedVectorCollection} to the {@link
 * StudioBackend} SPI and optionally fuses results with a {@link SidecartSource} for text and
 * full-text search. Supports tri-mode search (vector-only, text-only, hybrid via RRF) per the
 * distributed-studio plan §4.3.1.
 *
 * <p>The backend is constructed once per Studio process via {@link
 * #open(DistributedConnectionConfig)}; the sidecart is attached afterwards via {@link
 * #attachSidecart(SidecartSource)}, allowing the web layer to wire the same {@link SidecartSource}
 * both inside the backend and into its own registry for blob serving.
 */
public final class DistributedStudioBackend implements StudioBackend {

  private static final Logger LOG = LoggerFactory.getLogger(DistributedStudioBackend.class);

  private final String collectionName;
  private final DistributedVectorCollection collection;
  private final StorageBackend t3Backend;
  private final int dim;
  private final String metric;
  private final Instant createdAt;
  private final Map<String, Integer> idToOrdinal;
  private volatile SidecartSource sidecart;

  private DistributedStudioBackend(
      String collectionName,
      DistributedVectorCollection collection,
      StorageBackend t3Backend,
      int dim,
      String metric) {
    this.collectionName = collectionName;
    this.collection = collection;
    this.t3Backend = t3Backend;
    this.dim = dim;
    this.metric = metric;
    this.createdAt = Instant.now();
    this.idToOrdinal = new HashMap<>(collection.size());
    rebuildIdToOrdinal();
  }

  /**
   * Opens the collection from R2 + WAL using the supplied configuration. The returned backend owns
   * the {@link DistributedVectorCollection} and the underlying {@link S3StorageBackend}; both are
   * released via {@link #close()}.
   */
  public static DistributedStudioBackend open(DistributedConnectionConfig cfg) {
    Objects.requireNonNull(cfg, "cfg");
    StorageBackend t3 =
        S3StorageBackend.create(
            URI.create(cfg.s3Endpoint()),
            cfg.s3Bucket(),
            cfg.s3Region(),
            cfg.s3AccessKey(),
            cfg.s3SecretKey());
    StorageBackend prefixed =
        cfg.s3Prefix() == null || cfg.s3Prefix().isEmpty()
            ? t3
            : new PrefixedStorageBackend(t3, cfg.s3Prefix());
    try {
      DistributedVectorCollection col =
          DistributedVectorCollection.open(cfg.walDir(), cfg.metric(), cfg.tierPolicy(), prefixed);
      return new DistributedStudioBackend(
          cfg.collectionName(), col, prefixed, cfg.dim(), cfg.metric().name());
    } catch (IOException e) {
      throw new UncheckedIOException(
          "failed to open R2-backed collection " + cfg.collectionName(), e);
    }
  }

  /**
   * Attaches a sidecart for text/blob payloads and full-text search. Idempotent; the most recent
   * call wins. Pass {@code null} to detach.
   */
  public DistributedStudioBackend attachSidecart(SidecartSource source) {
    this.sidecart = source;
    return this;
  }

  // ─── StudioBackend ─────────────────────────────────────────────────────

  @Override
  public List<CollectionSummary> listCollections() {
    return List.of(summary());
  }

  @Override
  public CollectionSummary describe(String name) {
    require(name);
    return summary();
  }

  @Override
  public List<SearchHit> search(String name, SearchSpec spec) {
    require(name);
    Objects.requireNonNull(spec, "spec");
    boolean hasVec = spec.queryVector() != null && !isZeroVector(spec.queryVector());
    boolean hasText = spec.queryText() != null && !spec.queryText().isBlank();
    if (!hasVec && !hasText) {
      throw new IllegalArgumentException("queryVector or queryText required");
    }
    int k = spec.k();
    int overFetch = Math.max(k, k * 2);
    List<SearchHit> dense = hasVec ? denseSearch(spec.queryVector(), overFetch, spec) : List.of();
    List<TextSearchHit> sparse =
        hasText && sidecart != null ? sidecart.textSearch(spec.queryText(), overFetch) : List.of();
    if (!dense.isEmpty() && !sparse.isEmpty()) {
      return RrfFuser.fuse(dense, sparse, k, RrfFuser.DEFAULT_K_RRF, id -> hydrate(id, spec));
    }
    if (!sparse.isEmpty()) return materializeFromSparse(sparse, spec, k);
    return dense.size() > k ? dense.subList(0, k) : dense;
  }

  @Override
  public DocumentView getDocument(String name, String id) {
    require(name);
    Integer ord = idToOrdinal.get(id);
    if (ord == null) return null;
    float[] vec = spec_safeVectorAt(ord);
    SidecartRecord rec = sidecart == null ? null : sidecart.get(id).orElse(null);
    String text = rec == null ? null : rec.text();
    return new DocumentView(id, vec, text, null);
  }

  @Override
  public Optional<byte[]> getBlob(String name, String id) {
    require(name);
    if (sidecart == null) return Optional.empty();
    return sidecart.get(id).map(SidecartRecord::blob).filter(b -> b != null && b.length > 0);
  }

  @Override
  public List<DocumentView> previewDocuments(String name, int offset, int limit) {
    return documentPage(name, offset, limit).items();
  }

  @Override
  public DocumentPageView documentPage(String name, int offset, int limit) {
    require(name);
    if (offset < 0 || limit <= 0) {
      throw new IllegalArgumentException("offset>=0, limit>0");
    }
    List<String> ids = collection.allIdsView();
    int from = Math.min(offset, ids.size());
    int to = Math.min(from + limit, ids.size());
    List<String> page = ids.subList(from, to);
    Map<String, SidecartRecord> texts = sidecart == null ? Map.of() : sidecart.getAll(page);
    List<DocumentView> out = new ArrayList<>(page.size());
    for (int i = 0; i < page.size(); i++) {
      String id = page.get(i);
      Integer ord = idToOrdinal.get(id);
      float[] vec = ord == null ? null : spec_safeVectorAt(ord);
      SidecartRecord rec = texts.get(id);
      out.add(new DocumentView(id, vec, rec == null ? null : rec.text(), null));
    }
    return new DocumentPageView(out, ids.size());
  }

  @Override
  public float[][] vectorBatch(String name, List<String> ids) {
    require(name);
    Objects.requireNonNull(ids, "ids");
    List<float[]> out = new ArrayList<>(ids.size());
    for (String id : ids) {
      Integer ord = idToOrdinal.get(id);
      if (ord == null) continue;
      float[] v = spec_safeVectorAt(ord);
      if (v != null) out.add(v);
    }
    return out.toArray(new float[0][]);
  }

  @Override
  public void streamAllVectors(
      String name, BiConsumer<String, float[]> sink, IntConsumer progress) {
    require(name);
    Objects.requireNonNull(sink, "sink");
    List<String> ids = collection.allIdsView();
    int n = 0;
    for (String id : ids) {
      Integer ord = idToOrdinal.get(id);
      if (ord == null) continue;
      float[] v = spec_safeVectorAt(ord);
      if (v == null) continue;
      sink.accept(id, v);
      if (++n % 1024 == 0 && progress != null) progress.accept(n);
    }
    if (progress != null) progress.accept(n);
  }

  @Override
  public void commit(String name) {
    require(name);
    try {
      collection.commit();
    } catch (IOException e) {
      throw new UncheckedIOException("commit failed for " + name, e);
    }
    rebuildIdToOrdinal();
  }

  @Override
  public void deleteCollection(String name) {
    require(name);
    throw new UnsupportedOperationException(
        "DistributedStudioBackend does not support deleteCollection (DVC has no delete)");
  }

  @Override
  public void close() {
    try {
      collection.close();
    } catch (IOException e) {
      LOG.warn("error closing DistributedVectorCollection", e);
    }
    if (t3Backend instanceof AutoCloseable c) {
      try {
        c.close();
      } catch (Exception e) {
        LOG.warn("error closing t3 backend", e);
      }
    }
    if (sidecart != null) {
      try {
        sidecart.close();
      } catch (Exception e) {
        LOG.warn("error closing sidecart", e);
      }
    }
  }

  // ─── helpers ────────────────────────────────────────────────────────────

  private CollectionSummary summary() {
    return new CollectionSummary(
        collectionName, dim, metric, "IVF_DISTRIBUTED", "PQ", collection.size(), createdAt);
  }

  private void require(String name) {
    if (!collectionName.equals(name)) {
      throw new IllegalArgumentException("unknown collection: " + name);
    }
  }

  private void rebuildIdToOrdinal() {
    idToOrdinal.clear();
    List<String> ids = collection.allIdsView();
    for (int i = 0; i < ids.size(); i++) idToOrdinal.put(ids.get(i), i);
  }

  /** Defensive vector lookup: ordinals can shift after commits, so absent ordinals return null. */
  private float[] spec_safeVectorAt(int ord) {
    if (ord < 0 || ord >= collection.size()) return null;
    return collection.vectorAt(ord);
  }

  private static boolean isZeroVector(float[] v) {
    for (float x : v) if (x != 0f) return false;
    return true;
  }

  private List<SearchHit> denseSearch(float[] query, int k, SearchSpec spec) {
    int nprobe = Math.max(4, Math.min(32, (int) Math.ceil(Math.sqrt(collection.size()))));
    List<IvfHit> raw = collection.search(query, k, nprobe);
    if (raw.isEmpty()) return List.of();
    Map<String, SidecartRecord> texts = Map.of();
    if (spec.includeText() && sidecart != null) {
      List<String> ids = new ArrayList<>(raw.size());
      for (IvfHit h : raw) ids.add(h.id());
      texts = sidecart.getAll(ids);
    }
    List<SearchHit> out = new ArrayList<>(raw.size());
    for (IvfHit h : raw) {
      float[] vec = spec.includeVector() ? spec_safeVectorAt(h.ordinal()) : null;
      SidecartRecord rec = texts.get(h.id());
      String text = rec == null ? null : rec.text();
      out.add(new SearchHit(h.id(), h.score(), vec, text, null));
    }
    return out;
  }

  private List<SearchHit> materializeFromSparse(
      List<TextSearchHit> sparse, SearchSpec spec, int k) {
    int limit = Math.min(k, sparse.size());
    List<SearchHit> out = new ArrayList<>(limit);
    Map<String, SidecartRecord> texts = Map.of();
    if (spec.includeText() && sidecart != null) {
      List<String> ids = new ArrayList<>(limit);
      for (int i = 0; i < limit; i++) ids.add(sparse.get(i).id());
      texts = sidecart.getAll(ids);
    }
    for (int i = 0; i < limit; i++) {
      TextSearchHit h = sparse.get(i);
      Integer ord = idToOrdinal.get(h.id());
      float[] vec = spec.includeVector() && ord != null ? spec_safeVectorAt(ord) : null;
      SidecartRecord rec = texts.get(h.id());
      String text = rec == null ? null : rec.text();
      double score = h.score() / (1.0 + Math.abs(h.score()));
      out.add(new SearchHit(h.id(), score, vec, text, null));
    }
    return out;
  }

  /** Looks up vector + sidecart text for a sparse-only id during RRF fusion. */
  private SearchHit hydrate(String id, SearchSpec spec) {
    Integer ord = idToOrdinal.get(id);
    float[] vec = spec.includeVector() && ord != null ? spec_safeVectorAt(ord) : null;
    SidecartRecord rec =
        spec.includeText() && sidecart != null ? sidecart.get(id).orElse(null) : null;
    return new SearchHit(id, 0d, vec, rec == null ? null : rec.text(), null);
  }
}
