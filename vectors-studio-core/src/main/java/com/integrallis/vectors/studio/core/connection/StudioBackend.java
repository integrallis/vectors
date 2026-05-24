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

import com.integrallis.vectors.studio.core.search.DocumentPageView;
import com.integrallis.vectors.studio.core.search.DocumentView;
import com.integrallis.vectors.studio.core.search.SearchHit;
import com.integrallis.vectors.studio.core.search.SearchSpec;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.IntConsumer;

/**
 * UI-agnostic backend SPI exposed by Studio for any vector store: an embedded {@link
 * com.integrallis.vectors.db.VectorCollection}, a remote {@code vectors-server} reached over HTTP,
 * or any provider registered via {@link StudioBackendProvider}. Implementations are obtained
 * through {@link StudioBackendFactory}.
 */
public interface StudioBackend extends AutoCloseable {

  /** Returns metadata for every collection available through this backend. */
  List<CollectionSummary> listCollections();

  /** Describes a single collection by name. */
  CollectionSummary describe(String name);

  /** Runs a vector or hybrid search and returns ordered hits. */
  List<SearchHit> search(String name, SearchSpec spec);

  /** Returns a single document, or {@code null} if unknown. */
  DocumentView getDocument(String name, String id);

  /**
   * Returns the binary blob associated with a document (e.g. an extracted image), or {@link
   * Optional#empty()} if no blob is registered for that id. The embedded backend always returns
   * empty because it does not own the text/blob index; the remote backend proxies to the server's
   * {@code GET /v1/collections/{name}/blobs/{id}} endpoint.
   */
  Optional<byte[]> getBlob(String name, String id);

  /** Returns a paginated preview of live documents. */
  List<DocumentView> previewDocuments(String name, int offset, int limit);

  /**
   * Returns one page of live documents <i>and</i> the collection's live document count, in a single
   * call. Implementations should resolve both atomically when their underlying transport supports
   * it (the remote backend's {@code GET /documents} returns both).
   */
  DocumentPageView documentPage(String name, int offset, int limit);

  /** Bulk fetches raw vectors by id. The returned matrix excludes ids that did not resolve. */
  float[][] vectorBatch(String name, List<String> ids);

  /**
   * Streams every live (id, vector) in the collection through {@code sink}. {@code progress} is
   * called periodically with the running count of processed vectors.
   */
  void streamAllVectors(String name, BiConsumer<String, float[]> sink, IntConsumer progress);

  /** Forces any pending writes to be made visible (no-op for read-only remote backends). */
  void commit(String name);

  /**
   * Deletes a collection. For remote backends this proxies to the server's {@code DELETE
   * /v1/collections/{name}} endpoint; for embedded backends it closes the collection and removes it
   * from the in-process map. The on-disk storage directory is not removed by either implementation
   * (the embedded backend does not own the directory layout).
   */
  void deleteCollection(String name);

  /**
   * Applies a trial's optimized index parameters to the live collection by rebuilding it.
   *
   * <p>The trial's {@code params} map uses the axis names defined by the optimizer's search space —
   * the same names {@code IndexStudy} maps onto {@link
   * com.integrallis.vectors.db.VectorCollectionBuilder}: {@code metric}, {@code indexType}, {@code
   * quantizer}, and (for HNSW) {@code m}/{@code efConstruction}, or (for VAMANA) {@code
   * vamanaR}/{@code vamanaL}/{@code vamanaAlpha}. The collection's dimension is preserved
   * regardless of trial params. Search-time-only axes (e.g. {@code efSearch}) are accepted but do
   * not influence the rebuild; they are echoed in {@link ApplyTrialResult#appliedParams()}.
   *
   * <p>Implementations rebuild the collection by reading the live document set, constructing a new
   * collection from the merged config, ingesting every document, and atomically swapping the new
   * collection into the registry in place of the old one. The old collection is then closed.
   *
   * <p>The default implementation throws {@link UnsupportedOperationException}: backends that
   * cannot reconfigure a live collection (e.g. the remote backend, which would require a
   * server-side rebuild endpoint) opt out by not overriding.
   *
   * @param name the collection to reconfigure
   * @param trialParams the trial's parameter map
   * @return a summary of what was applied
   * @throws IllegalArgumentException if the collection is unknown
   * @throws UnsupportedOperationException if this backend cannot reconfigure live collections
   */
  default ApplyTrialResult applyTrialParameters(String name, Map<String, Object> trialParams) {
    throw new UnsupportedOperationException(
        "applyTrialParameters is not supported by this backend ("
            + getClass().getSimpleName()
            + "); use an embedded backend to reconfigure a live collection.");
  }

  @Override
  void close();
}
