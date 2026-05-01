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

  /** Releases any handles held by the source. The default is a no-op. */
  @Override
  default void close() {}
}
