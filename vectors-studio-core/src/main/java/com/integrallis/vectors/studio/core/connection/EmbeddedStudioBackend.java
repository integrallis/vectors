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
package com.integrallis.vectors.studio.core.connection;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.filter.Filter;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.VectorCollectionBuilder;
import com.integrallis.vectors.db.VectorCollectionConfig;
import com.integrallis.vectors.db.storage.FileFormat;
import com.integrallis.vectors.db.storage.GenerationDirectory;
import com.integrallis.vectors.db.storage.Manifest;
import com.integrallis.vectors.studio.core.search.DocumentPageView;
import com.integrallis.vectors.studio.core.search.DocumentView;
import com.integrallis.vectors.studio.core.search.SearchHit;
import com.integrallis.vectors.studio.core.search.SearchSpec;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiConsumer;
import java.util.function.IntConsumer;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Backend that adapts a set of in-process {@link VectorCollection} instances to the {@link
 * StudioBackend} SPI. Mirrors the discovery loop in {@code
 * com.integrallis.vectors.server.CollectionDiscovery}.
 */
public final class EmbeddedStudioBackend implements StudioBackend {

  private static final Logger LOG = LoggerFactory.getLogger(EmbeddedStudioBackend.class);
  private static final Pattern NAME_PATTERN = Pattern.compile("^[A-Za-z0-9_-]{1,128}$");

  private final ConcurrentHashMap<String, VectorCollection> open;
  private final ConcurrentHashMap<String, Instant> createdAt;
  // Names whose collection was originally opened from a persistent directory. Tracked so that a
  // subsequent in-memory rebuild via applyTrialParameters can honestly report that the on-disk
  // image has diverged from the live state.
  private final java.util.Set<String> persistentNames = ConcurrentHashMap.newKeySet();

  private EmbeddedStudioBackend(
      ConcurrentHashMap<String, VectorCollection> open,
      ConcurrentHashMap<String, Instant> createdAt) {
    this.open = open;
    this.createdAt = createdAt;
  }

  /**
   * Opens by scanning {@code dataDir} for persisted collections. Empty/null dir yields no entries.
   */
  public static EmbeddedStudioBackend open(Path dataDir) {
    ConcurrentHashMap<String, VectorCollection> open = new ConcurrentHashMap<>();
    ConcurrentHashMap<String, Instant> ts = new ConcurrentHashMap<>();
    java.util.Set<String> persistentNamesAtBoot = new java.util.HashSet<>();
    if (dataDir == null || !Files.isDirectory(dataDir)) {
      return new EmbeddedStudioBackend(open, ts);
    }
    try (Stream<Path> subdirs = Files.list(dataDir)) {
      for (Path candidate : (Iterable<Path>) subdirs::iterator) {
        if (!Files.isDirectory(candidate)) continue;
        Path fn = candidate.getFileName();
        if (fn == null) continue;
        String name = fn.toString();
        if (!NAME_PATTERN.matcher(name).matches()) continue;
        try {
          long gen = GenerationDirectory.readCurrent(candidate);
          if (gen < 0) continue;
          Path manifest =
              candidate
                  .resolve(FileFormat.generationDirName(gen))
                  .resolve(FileFormat.MANIFEST_FILE);
          if (!Files.exists(manifest)) continue;
          Manifest m = Manifest.readFrom(manifest);
          VectorCollection c =
              VectorCollection.builder()
                  .dimension(m.dimension())
                  .metric(m.metric())
                  .indexType(m.indexType())
                  .quantizer(m.quantizerKind())
                  .storagePath(candidate)
                  .build();
          open.put(name, c);
          ts.put(name, readCreationInstant(candidate, m));
          // Reopened-from-disk means originally persistent — tracked so applyTrialParameters can
          // honestly report that a subsequent in-memory rebuild has diverged from disk.
          persistentNamesAtBoot.add(name);
        } catch (RuntimeException | IOException e) {
          LOG.warn("studio: failed to reopen {}: {}", name, e.getMessage());
        }
      }
    } catch (IOException e) {
      LOG.warn("studio: failed to scan {}: {}", dataDir, e.getMessage());
    }
    EmbeddedStudioBackend backend = new EmbeddedStudioBackend(open, ts);
    backend.persistentNames.addAll(persistentNamesAtBoot);
    return backend;
  }

  /** Test/seam factory: wraps a pre-built map of collections. */
  public static EmbeddedStudioBackend withCollections(Map<String, VectorCollection> collections) {
    Objects.requireNonNull(collections, "collections");
    ConcurrentHashMap<String, VectorCollection> map = new ConcurrentHashMap<>(collections);
    ConcurrentHashMap<String, Instant> ts = new ConcurrentHashMap<>();
    Instant now = Instant.now();
    for (String n : map.keySet()) ts.put(n, now);
    return new EmbeddedStudioBackend(map, ts);
  }

  /** Adapts an additional in-memory {@link VectorCollection} into this backend. */
  public EmbeddedStudioBackend register(String name, VectorCollection collection) {
    Objects.requireNonNull(collection, "collection");
    if (!NAME_PATTERN.matcher(name).matches()) {
      throw new IllegalArgumentException("invalid collection name: " + name);
    }
    open.put(name, collection);
    createdAt.put(name, Instant.now());
    return this;
  }

  @Override
  public void addCollection(String name, VectorCollection collection) {
    Objects.requireNonNull(collection, "collection");
    if (!NAME_PATTERN.matcher(name).matches()) {
      throw new IllegalArgumentException("invalid collection name: " + name);
    }
    if (open.containsKey(name)) {
      throw new IllegalArgumentException("collection already exists: " + name);
    }
    open.put(name, collection);
    createdAt.put(name, Instant.now());
  }

  @Override
  public List<CollectionSummary> listCollections() {
    List<CollectionSummary> out = new ArrayList<>(open.size());
    for (Map.Entry<String, VectorCollection> e : open.entrySet()) {
      out.add(toSummary(e.getKey(), e.getValue()));
    }
    out.sort((a, b) -> a.name().compareTo(b.name()));
    return out;
  }

  @Override
  public CollectionSummary describe(String name) {
    VectorCollection c = require(name);
    return toSummary(name, c);
  }

  @Override
  public List<SearchHit> search(String name, SearchSpec spec) {
    VectorCollection c = require(name);
    Objects.requireNonNull(spec, "spec");
    if (spec.queryVector() == null) {
      throw new IllegalArgumentException("EmbeddedStudioBackend requires queryVector");
    }
    SearchRequest.Builder b =
        SearchRequest.builder(spec.queryVector(), spec.k())
            .includeVector(spec.includeVector())
            .includeText(spec.includeText())
            .includeMetadata(spec.includeMetadata());
    Filter filter = (Filter) (spec.filter() == null ? null : spec.filter().get("__filter__"));
    if (filter != null) b.filter(filter);
    SearchResult r = c.search(b.build());
    List<SearchHit> hits = new ArrayList<>(r.hits().size());
    for (SearchResult.Hit h : r.hits()) {
      Document d = h.document();
      hits.add(
          new SearchHit(
              h.id(),
              h.score(),
              spec.includeVector() ? d.vector() : null,
              spec.includeText() ? d.text() : null,
              spec.includeMetadata() ? MetadataAdapter.toMap(d.metadata()) : null));
    }
    return hits;
  }

  @Override
  public DocumentView getDocument(String name, String id) {
    VectorCollection c = require(name);
    Document d = c.get(id);
    if (d == null) return null;
    return toView(d);
  }

  @Override
  public Optional<byte[]> getBlob(String name, String id) {
    require(name);
    return Optional.empty();
  }

  @Override
  public List<DocumentView> previewDocuments(String name, int offset, int limit) {
    return documentPage(name, offset, limit).items();
  }

  @Override
  public DocumentPageView documentPage(String name, int offset, int limit) {
    VectorCollection c = require(name);
    if (offset < 0 || limit <= 0) {
      throw new IllegalArgumentException("offset>=0, limit>0");
    }
    List<Document> all = c.documents();
    int from = Math.min(offset, all.size());
    int to = Math.min(from + limit, all.size());
    List<DocumentView> out = new ArrayList<>(to - from);
    for (int i = from; i < to; i++) out.add(toView(all.get(i)));
    return new DocumentPageView(out, all.size());
  }

  @Override
  public float[][] vectorBatch(String name, List<String> ids) {
    VectorCollection c = require(name);
    Objects.requireNonNull(ids, "ids");
    List<float[]> hits = new ArrayList<>(ids.size());
    for (String id : ids) {
      Document d = c.get(id);
      if (d != null && d.vector() != null) hits.add(d.vector());
    }
    return hits.toArray(new float[0][]);
  }

  @Override
  public void streamAllVectors(
      String name, BiConsumer<String, float[]> sink, IntConsumer progress) {
    VectorCollection c = require(name);
    Objects.requireNonNull(sink, "sink");
    int n = 0;
    for (Document d : c.documents()) {
      if (d.vector() == null) continue;
      sink.accept(d.id(), d.vector());
      if (++n % 1024 == 0 && progress != null) progress.accept(n);
    }
    if (progress != null) progress.accept(n);
  }

  @Override
  public void commit(String name) {
    require(name).commit();
  }

  @Override
  public void deleteCollection(String name) {
    VectorCollection c = open.remove(name);
    createdAt.remove(name);
    if (c == null) throw new IllegalArgumentException("unknown collection: " + name);
    try {
      c.close();
    } catch (RuntimeException e) {
      LOG.warn("studio: failed to close removed collection {}: {}", name, e.getMessage());
    }
  }

  @Override
  public ApplyTrialResult applyTrialParameters(String name, Map<String, Object> trialParams) {
    Objects.requireNonNull(trialParams, "trialParams");
    long start = System.nanoTime();
    // computeIfPresent so the swap is atomic against concurrent reads/writes on the same name in
    // the ConcurrentHashMap; competing applyTrialParameters / deleteCollection for the same name
    // are serialised by ConcurrentHashMap's per-bin lock.
    var swap = new java.util.concurrent.atomic.AtomicReference<ApplyTrialResult>();
    VectorCollection updated =
        open.computeIfPresent(
            name,
            (key, oldCol) -> {
              VectorCollectionConfig oldCfg = oldCol.config();
              List<Document> snapshot = oldCol.documents();
              Map<String, Object> resolved = resolveTrialParams(oldCfg, trialParams);
              VectorCollection rebuilt = buildFromResolved(oldCfg.dimension(), resolved);
              try {
                if (!snapshot.isEmpty()) {
                  rebuilt.addAll(snapshot);
                  rebuilt.commit();
                }
              } catch (RuntimeException e) {
                try {
                  rebuilt.close();
                } catch (RuntimeException closeEx) {
                  e.addSuppressed(closeEx);
                }
                throw e;
              }
              // Close old collection AFTER the new one is fully populated, so a build failure
              // leaves the old collection installed.
              try {
                oldCol.close();
              } catch (RuntimeException e) {
                LOG.warn(
                    "studio: failed to close old collection during applyTrialParameters({}): {}",
                    name,
                    e.getMessage());
              }
              long rebuildMs = (System.nanoTime() - start) / 1_000_000L;
              boolean wasPersistent = persistentNames.remove(name);
              swap.set(new ApplyTrialResult(resolved, snapshot.size(), rebuildMs, !wasPersistent));
              return rebuilt;
            });
    if (updated == null) {
      throw new IllegalArgumentException("unknown collection: " + name);
    }
    return swap.get();
  }

  /**
   * Resolves a trial's param map into the concrete collection-config knobs that will actually be
   * applied, falling back to the OLD collection's config for any axis the trial didn't sample.
   * Mirrors the axis names {@code IndexStudy.configureBuilder} understands.
   */
  private static Map<String, Object> resolveTrialParams(
      VectorCollectionConfig oldCfg, Map<String, Object> trialParams) {
    Map<String, Object> resolved = new LinkedHashMap<>();
    SimilarityFunction metric =
        enumOf(trialParams, "metric", SimilarityFunction.class, oldCfg.metric());
    IndexType indexType = enumOf(trialParams, "indexType", IndexType.class, oldCfg.indexType());
    QuantizerKind quantizer =
        enumOf(trialParams, "quantizer", QuantizerKind.class, oldCfg.quantizerKind());
    resolved.put("metric", metric.name());
    resolved.put("indexType", indexType.name());
    resolved.put("quantizer", quantizer.name());
    if (indexType == IndexType.HNSW) {
      int m = optInt(trialParams, "m", VectorCollectionBuilder.DEFAULT_HNSW_M);
      int efC =
          optInt(
              trialParams, "efConstruction", VectorCollectionBuilder.DEFAULT_HNSW_EF_CONSTRUCTION);
      resolved.put("m", m);
      resolved.put("efConstruction", efC);
    } else if (indexType == IndexType.VAMANA) {
      int r = optInt(trialParams, "vamanaR", VectorCollectionBuilder.DEFAULT_VAMANA_R);
      int l = optInt(trialParams, "vamanaL", VectorCollectionBuilder.DEFAULT_VAMANA_L);
      double alpha =
          optDouble(trialParams, "vamanaAlpha", VectorCollectionBuilder.DEFAULT_VAMANA_ALPHA);
      resolved.put("vamanaR", r);
      resolved.put("vamanaL", l);
      resolved.put("vamanaAlpha", alpha);
    }
    // Search-time-only axes are not used in rebuild but are echoed back in the result so the UI
    // can surface them; record raw values without re-validating.
    if (trialParams.containsKey("efSearch")) {
      resolved.put("efSearch", trialParams.get("efSearch"));
    }
    return resolved;
  }

  private static VectorCollection buildFromResolved(int dim, Map<String, Object> resolved) {
    SimilarityFunction metric = SimilarityFunction.valueOf((String) resolved.get("metric"));
    IndexType indexType = IndexType.valueOf((String) resolved.get("indexType"));
    QuantizerKind quantizer = QuantizerKind.valueOf((String) resolved.get("quantizer"));
    VectorCollectionBuilder b =
        VectorCollection.builder()
            .dimension(dim)
            .metric(metric)
            .indexType(indexType)
            .quantizer(quantizer);
    if (indexType == IndexType.HNSW) {
      b.hnswM((Integer) resolved.get("m"))
          .hnswEfConstruction((Integer) resolved.get("efConstruction"));
    } else if (indexType == IndexType.VAMANA) {
      b.vamanaMaxDegree((Integer) resolved.get("vamanaR"))
          .vamanaSearchListSize((Integer) resolved.get("vamanaL"))
          .vamanaAlpha(((Number) resolved.get("vamanaAlpha")).floatValue());
    }
    return b.build();
  }

  private static int optInt(Map<String, Object> p, String key, int defaultValue) {
    Object v = p.get(key);
    if (v == null) return defaultValue;
    return ((Number) v).intValue();
  }

  private static double optDouble(Map<String, Object> p, String key, double defaultValue) {
    Object v = p.get(key);
    if (v == null) return defaultValue;
    return ((Number) v).doubleValue();
  }

  private static <E extends Enum<E>> E enumOf(
      Map<String, Object> p, String key, Class<E> type, E defaultValue) {
    Object v = p.get(key);
    if (v == null) return defaultValue;
    if (type.isInstance(v)) return type.cast(v);
    return Enum.valueOf(type, v.toString().toUpperCase(Locale.ROOT));
  }

  @Override
  public void close() {
    for (VectorCollection c : open.values()) {
      try {
        c.close();
      } catch (RuntimeException e) {
        LOG.warn("studio: failed to close collection: {}", e.getMessage());
      }
    }
    open.clear();
  }

  // ---------- helpers ----------

  private VectorCollection require(String name) {
    VectorCollection c = open.get(name);
    if (c == null) throw new IllegalArgumentException("unknown collection: " + name);
    return c;
  }

  private CollectionSummary toSummary(String name, VectorCollection c) {
    var cfg = c.config();
    return new CollectionSummary(
        name,
        cfg.dimension(),
        cfg.metric().name(),
        cfg.indexType().name(),
        cfg.quantizerKind().name(),
        c.size(),
        createdAt.getOrDefault(name, Instant.EPOCH));
  }

  private static DocumentView toView(Document d) {
    return new DocumentView(d.id(), d.vector(), d.text(), MetadataAdapter.toMap(d.metadata()));
  }

  private static Instant readCreationInstant(Path storageRoot, Manifest m) {
    try {
      BasicFileAttributes attrs = Files.readAttributes(storageRoot, BasicFileAttributes.class);
      long t = attrs.creationTime().toMillis();
      if (t > 0) return Instant.ofEpochMilli(t);
    } catch (IOException ignored) {
    }
    return Instant.ofEpochMilli(m.createdEpochMillis());
  }
}
