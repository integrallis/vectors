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
package com.integrallis.vectors.studio.core.metadata;

import java.util.List;
import java.util.Optional;

/** Studio-side overlay schema for a collection's metadata fields. */
public record MetadataSchema(List<FieldSpec> fields) {

  /** Empty schema. */
  public static MetadataSchema empty() {
    return new MetadataSchema(List.of());
  }

  /** Returns the field spec with the given name, if any. */
  public Optional<FieldSpec> find(String name) {
    return fields.stream().filter(f -> f.name().equals(name)).findFirst();
  }
}
