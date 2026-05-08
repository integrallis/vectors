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
package com.integrallis.vectors.studio.sidecart;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * SPI for an external (sidecart) data source that holds the non-vector payload of documents — text,
 * images, audio, or arbitrary binary — keyed by the same external document id used by the
 * vectors-server collection. The Studio web layer consults a registered source whenever the
 * collection itself does not carry the desired payload, enabling thin vector indexes that point at
 * H2, the filesystem, or a remote object store.
 *
 * <p>Implementations are expected to be thread-safe and inexpensive to call repeatedly: Studio
 * surfaces sidecart contents both in document-detail pages and as inline image responses through
 * the {@code /collections/{name}/blobs/{id}} route.
 */
public interface SidecartSource extends AutoCloseable {

  /**
   * Returns the sidecart record for {@code id}, or {@link Optional#empty()} if the source has no
   * row matching that id. Implementations should map authentic absence to {@code empty()} and raise
   * {@link SidecartSourceException} only for transport / parse failures.
   */
  Optional<SidecartRecord> get(String id);

  /**
   * Batch read for {@code ids}. The default fans out to {@link #get(String)} per id; backends where
   * round-trip latency dominates (D1, S3, REST) should override this with a single {@code WHERE id
   * IN (…)} round-trip. Missing ids are simply absent from the returned map.
   */
  default Map<String, SidecartRecord> getAll(Collection<String> ids) {
    Map<String, SidecartRecord> out = new HashMap<>(ids.size());
    for (String id : ids) {
      get(id).ifPresent(r -> out.put(id, r));
    }
    return out;
  }

  /**
   * Lexical / full-text search over the sidecart's text payload. The default returns an empty list,
   * meaning "this sidecart does not provide FTS"; backends that wrap a SQL engine with FTS (H2
   * native fulltext, SQLite FTS5 over D1) override this. Hits are returned in score-descending
   * order.
   *
   * @param query free-text query in the engine-native syntax
   * @param k maximum hits to return
   */
  default List<TextSearchHit> textSearch(String query, int k) {
    return List.of();
  }

  /** Releases any handles held by the source. The default is a no-op. */
  @Override
  default void close() {}
}
