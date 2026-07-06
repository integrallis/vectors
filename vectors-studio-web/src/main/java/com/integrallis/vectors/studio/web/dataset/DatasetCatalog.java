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
package com.integrallis.vectors.studio.web.dataset;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Loads the bundled {@code sample-datasets.json} catalog from the classpath into a list of {@link
 * DatasetCatalogEntry}. If the environment variable {@code VECTORS_STUDIO_DATASETS} points at a
 * readable JSON file with the same schema, its entries are merged in by {@code id} (user entries
 * override bundled ones and may add new ids). The catalog is immutable once constructed.
 */
public final class DatasetCatalog {

  /** Env var pointing at a user-supplied catalog file to merge over the bundled defaults. */
  public static final String OVERRIDE_ENV = "VECTORS_STUDIO_DATASETS";

  private static final Logger LOG = LoggerFactory.getLogger(DatasetCatalog.class);
  private static final String BUNDLED_RESOURCE = "/sample-datasets.json";
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private final List<DatasetCatalogEntry> entries;
  private final Map<String, DatasetCatalogEntry> byId;

  private DatasetCatalog(List<DatasetCatalogEntry> entries) {
    this.entries = List.copyOf(entries);
    Map<String, DatasetCatalogEntry> index = new LinkedHashMap<>();
    for (DatasetCatalogEntry e : entries) {
      index.put(e.id(), e);
    }
    this.byId = Collections.unmodifiableMap(index);
  }

  /** Loads the bundled catalog and merges the optional {@code VECTORS_STUDIO_DATASETS} override. */
  public static DatasetCatalog load() {
    return load(System.getenv(OVERRIDE_ENV));
  }

  /**
   * Loads the bundled catalog and merges the override file at {@code overridePath} (may be null).
   */
  public static DatasetCatalog load(String overridePath) {
    // Preserve declaration order; a later entry with the same id replaces the earlier one.
    Map<String, DatasetCatalogEntry> merged = new LinkedHashMap<>();
    for (DatasetCatalogEntry e : readBundled()) {
      merged.put(e.id(), e);
    }
    if (overridePath != null && !overridePath.isBlank()) {
      for (DatasetCatalogEntry e : readOverride(Path.of(overridePath.trim()))) {
        merged.put(e.id(), e);
      }
    }
    return new DatasetCatalog(new ArrayList<>(merged.values()));
  }

  private static List<DatasetCatalogEntry> readBundled() {
    try (InputStream in = DatasetCatalog.class.getResourceAsStream(BUNDLED_RESOURCE)) {
      if (in == null) {
        LOG.warn("studio: bundled dataset catalog {} not found on classpath", BUNDLED_RESOURCE);
        return List.of();
      }
      return parse(MAPPER.readTree(in));
    } catch (IOException e) {
      LOG.warn("studio: failed to read bundled dataset catalog: {}", e.getMessage());
      return List.of();
    }
  }

  private static List<DatasetCatalogEntry> readOverride(Path path) {
    if (!Files.isReadable(path)) {
      LOG.warn("studio: {} points at unreadable file {}", OVERRIDE_ENV, path);
      return List.of();
    }
    try (InputStream in = Files.newInputStream(path)) {
      List<DatasetCatalogEntry> extra = parse(MAPPER.readTree(in));
      LOG.info("studio: merged {} dataset entries from {}", extra.size(), path);
      return extra;
    } catch (IOException e) {
      LOG.warn("studio: failed to read {} file {}: {}", OVERRIDE_ENV, path, e.getMessage());
      return List.of();
    }
  }

  private static List<DatasetCatalogEntry> parse(JsonNode root) {
    JsonNode datasets = root == null ? null : root.get("datasets");
    if (datasets == null || !datasets.isArray()) {
      return List.of();
    }
    List<DatasetCatalogEntry> out = new ArrayList<>(datasets.size());
    for (JsonNode n : datasets) {
      String id = text(n, "id", null);
      if (id == null || id.isBlank()) {
        continue; // an entry without an id cannot be a collection name; skip defensively
      }
      out.add(
          new DatasetCatalogEntry(
              id,
              text(n, "name", id),
              text(n, "description", ""),
              text(n, "domain", ""),
              text(n, "model", ""),
              n.path("dimension").asInt(0),
              text(n, "hfDataset", null),
              text(n, "config", "default"),
              text(n, "split", "train"),
              text(n, "vectorColumn", null),
              text(n, "textColumn", null),
              text(n, "idColumn", null),
              text(n, "metric", "COSINE"),
              n.path("defaultLimit").asInt(2000)));
    }
    return out;
  }

  private static String text(JsonNode n, String field, String dflt) {
    JsonNode v = n.get(field);
    return v == null || v.isNull() ? dflt : v.asText();
  }

  /** All catalog entries, in merged declaration order. */
  public List<DatasetCatalogEntry> entries() {
    return entries;
  }

  /** Looks up an entry by id (= collection name). */
  public Optional<DatasetCatalogEntry> byId(String id) {
    return Optional.ofNullable(byId.get(id));
  }

  /** The set of catalog ids, used to decorate matching collections as "sample" on the list page. */
  public Set<String> ids() {
    return byId.keySet();
  }
}
