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

import com.integrallis.vectors.server.client.CollectionInfo;
import com.integrallis.vectors.server.client.DocumentPage;
import com.integrallis.vectors.server.client.SampleResponse;
import com.integrallis.vectors.server.client.VectorsBatchResponse;
import com.integrallis.vectors.server.client.VectorsServerClient;
import com.integrallis.vectors.studio.core.search.DocumentPageView;
import com.integrallis.vectors.studio.core.search.DocumentView;
import com.integrallis.vectors.studio.core.search.SearchHit;
import com.integrallis.vectors.studio.core.search.SearchSpec;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.IntConsumer;

/** {@link StudioBackend} implementation that proxies to a remote {@code vectors-server}. */
public final class RemoteStudioBackend implements StudioBackend {

  private static final int PAGE_SIZE = 1024;

  private final VectorsServerClient client;

  private RemoteStudioBackend(VectorsServerClient client) {
    this.client = Objects.requireNonNull(client, "client");
  }

  /** Opens a remote backend from a {@link ConnectionConfig.Remote} configuration. */
  public static RemoteStudioBackend open(ConnectionConfig.Remote cfg) {
    Objects.requireNonNull(cfg, "cfg");
    return new RemoteStudioBackend(
        new VectorsServerClient(cfg.baseUrl().toString(), cfg.timeout(), cfg.token()));
  }

  @Override
  public List<CollectionSummary> listCollections() {
    List<CollectionInfo> infos = client.listCollections();
    List<CollectionSummary> out = new ArrayList<>(infos.size());
    for (CollectionInfo i : infos) out.add(toSummary(i));
    return out;
  }

  @Override
  public CollectionSummary describe(String name) {
    return toSummary(client.describe(name));
  }

  @Override
  public List<SearchHit> search(String name, SearchSpec spec) {
    Objects.requireNonNull(spec, "spec");
    if (spec.queryVector() == null && spec.queryText() == null) {
      throw new IllegalArgumentException("either queryVector or queryText must be supplied");
    }
    List<com.integrallis.vectors.server.client.SearchHit> raw;
    if (spec.queryText() != null && !spec.queryText().isBlank()) {
      raw = client.hybridSearch(name, spec.queryVector(), spec.queryText(), spec.k(), null);
    } else {
      raw = client.search(name, spec.queryVector(), spec.k(), null, spec.filter());
    }
    List<SearchHit> out = new ArrayList<>(raw.size());
    for (var h : raw) {
      out.add(
          new SearchHit(
              h.id(),
              h.score(),
              spec.includeVector() ? h.vector() : null,
              spec.includeText() ? h.text() : null,
              spec.includeMetadata() ? h.metadata() : null));
    }
    return out;
  }

  @Override
  public void deleteCollection(String name) {
    client.deleteCollection(name);
  }

  @Override
  public DocumentView getDocument(String name, String id) {
    Optional<DocumentPage.Item> opt = client.getDocument(name, id);
    if (opt.isEmpty()) return null;
    DocumentPage.Item it = opt.get();
    return new DocumentView(
        it.id(), it.vector(), it.text(), MetadataAdapter.fromJsonNode(it.metadata()));
  }

  @Override
  public Optional<byte[]> getBlob(String name, String id) {
    return client.getBlob(name, id);
  }

  @Override
  public List<DocumentView> previewDocuments(String name, int offset, int limit) {
    return documentPage(name, offset, limit).items();
  }

  @Override
  public DocumentPageView documentPage(String name, int offset, int limit) {
    DocumentPage page = client.previewDocuments(name, offset, limit, true);
    List<DocumentView> out = new ArrayList<>(page.items().size());
    for (DocumentPage.Item it : page.items()) {
      out.add(
          new DocumentView(
              it.id(), it.vector(), it.text(), MetadataAdapter.fromJsonNode(it.metadata())));
    }
    return new DocumentPageView(out, page.total());
  }

  @Override
  public float[][] vectorBatch(String name, List<String> ids) {
    VectorsBatchResponse r = client.vectorsBatch(name, ids);
    return r.vectors();
  }

  @Override
  public void streamAllVectors(
      String name, BiConsumer<String, float[]> sink, IntConsumer progress) {
    Objects.requireNonNull(sink, "sink");
    int offset = 0;
    int total = 0;
    while (true) {
      DocumentPage page = client.previewDocuments(name, offset, PAGE_SIZE, true);
      if (page.items().isEmpty()) break;
      for (DocumentPage.Item it : page.items()) {
        if (it.vector() != null) {
          sink.accept(it.id(), it.vector());
          total++;
        }
      }
      if (progress != null) progress.accept(total);
      offset += page.items().size();
      if (offset >= page.total()) break;
    }
  }

  /** Server-side {@code SampleResponse} pass-through (used by the projector). */
  public SampleResponse sample(String name, int n, boolean includeMetadata) {
    return client.sample(name, n, includeMetadata);
  }

  @Override
  public void commit(String name) {
    client.commit(name);
  }

  @Override
  public void close() {
    client.close();
  }

  private static CollectionSummary toSummary(CollectionInfo i) {
    return new CollectionSummary(
        i.name(), i.dimension(), i.metric(), i.indexType(), i.quantizer(), i.size(), i.createdAt());
  }
}
