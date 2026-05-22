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
package com.integrallis.vectors.optimizer.data;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.db.VectorCollection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Derives "same metadata label = relevant" qrels from a document collection. Useful when a dataset
 * has a category/intent column but lacks explicit relevance judgements.
 *
 * <p>{@link MetadataValue.Str} fields contribute one label per doc; {@link MetadataValue.Tags}
 * fields contribute the doc to every tag in the list. Other {@link MetadataValue} subtypes are
 * counted as unlabeled.
 */
public final class MetadataQrelsDeriver {

  /** Coverage report returned alongside the {@link Qrels}. */
  public record Coverage(int labeledDocs, int unlabeledDocs, int distinctLabels) {

    public int total() {
      return labeledDocs + unlabeledDocs;
    }
  }

  /** {@link Qrels} plus the coverage report describing how the labels were extracted. */
  public record Result(Qrels qrels, Coverage coverage) {}

  private MetadataQrelsDeriver() {}

  /** Derives qrels from a {@link VectorCollection}; iterates {@code col.documents()}. */
  public static Result derive(VectorCollection col, String metadataField) {
    Objects.requireNonNull(col, "col");
    return derive(col.documents(), metadataField);
  }

  /** Derives qrels from an in-memory list of documents. */
  public static Result derive(Iterable<Document> docs, String metadataField) {
    Objects.requireNonNull(docs, "docs");
    Objects.requireNonNull(metadataField, "metadataField");

    Map<String, Set<String>> labelToDocIds = new LinkedHashMap<>();
    int labeled = 0;
    int unlabeled = 0;
    for (Document d : docs) {
      List<String> labels = labelsOf(d, metadataField);
      if (labels.isEmpty()) {
        unlabeled++;
        continue;
      }
      labeled++;
      for (String label : labels) {
        labelToDocIds.computeIfAbsent(label, k -> new LinkedHashSet<>()).add(d.id());
      }
    }
    int total = labeled + unlabeled;
    if (total > 0 && (double) unlabeled / total > 0.5) {
      throw new IllegalStateException(
          "Metadata field '"
              + metadataField
              + "' covers only "
              + labeled
              + "/"
              + total
              + " docs; refusing to derive qrels with >50% unlabeled");
    }

    // Build qrels: for each labeled doc, every doc with the same label is relevant=1.
    Map<String, Map<String, Integer>> rel = new LinkedHashMap<>();
    for (Document d : docs) {
      List<String> labels = labelsOf(d, metadataField);
      if (labels.isEmpty()) continue;
      Map<String, Integer> row = new LinkedHashMap<>();
      for (String label : labels) {
        Set<String> peers = labelToDocIds.getOrDefault(label, Set.of());
        for (String peerId : peers) {
          if (!peerId.equals(d.id())) row.put(peerId, 1);
        }
      }
      if (!row.isEmpty()) rel.put(d.id(), row);
    }
    return new Result(new Qrels(rel), new Coverage(labeled, unlabeled, labelToDocIds.size()));
  }

  private static List<String> labelsOf(Document d, String field) {
    MetadataValue v = d.metadata().get(field);
    if (v == null) return List.of();
    return switch (v) {
      case MetadataValue.Str s -> List.of(s.value());
      case MetadataValue.Tags t -> List.copyOf(t.values());
      case MetadataValue.Num ignored -> List.of();
      case MetadataValue.Bool ignored -> List.of();
    };
  }
}
