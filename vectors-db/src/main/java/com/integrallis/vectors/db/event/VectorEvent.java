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
package com.integrallis.vectors.db.event;

import com.integrallis.vectors.db.MetadataValue;
import java.util.Map;
import java.util.Objects;

/**
 * Sealed event hierarchy for the vector collection mutation log. Each event corresponds to a single
 * mutation applied to a {@link com.integrallis.vectors.db.VectorCollection}.
 *
 * <p>Events are written to a {@link com.integrallis.vectors.storage.wal.SegmentedWriteAheadLog} via
 * {@link VectorEventCodec} and can be replayed to reconstruct collection state.
 */
public sealed interface VectorEvent
    permits VectorEvent.Add, VectorEvent.Delete, VectorEvent.Upsert, VectorEvent.Commit {

  /**
   * A new document was staged for insertion. The document id must not exist in the live generation
   * at the time of staging.
   */
  record Add(String id, float[] vector, String text, Map<String, MetadataValue> metadata)
      implements VectorEvent {
    public Add {
      Objects.requireNonNull(id, "id");
      Objects.requireNonNull(vector, "vector");
      metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }
  }

  /** A tombstone was staged for the document with the given id. Takes effect on the next commit. */
  record Delete(String id) implements VectorEvent {
    public Delete {
      Objects.requireNonNull(id, "id");
    }
  }

  /**
   * A document was inserted-or-replaced. Equivalent to delete-then-add but expressed as a single
   * atomic intent.
   */
  record Upsert(String id, float[] vector, String text, Map<String, MetadataValue> metadata)
      implements VectorEvent {
    public Upsert {
      Objects.requireNonNull(id, "id");
      Objects.requireNonNull(vector, "vector");
      metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }
  }

  /**
   * A generation was committed. Records the resulting generation number so that a replaying node
   * can reconstruct commit boundaries.
   */
  record Commit(long generationNumber) implements VectorEvent {}
}
