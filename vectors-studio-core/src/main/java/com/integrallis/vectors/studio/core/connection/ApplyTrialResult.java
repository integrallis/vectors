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

import java.util.Map;

/**
 * Outcome of {@link StudioBackend#applyTrialParameters(String, Map)}: the parameters that were
 * actually carried into the new collection config, the number of documents copied across the
 * rebuild, the build wall-time, and a flag indicating whether the underlying storage was kept in
 * sync with the rebuild. For embedded in-memory collections {@code persistenceRefreshed} is
 * trivially {@code true}; for embedded collections that were originally opened from a persistent
 * directory the in-memory image is now diverged from disk and the flag is {@code false} so callers
 * can surface that to the operator.
 *
 * @param appliedParams the resolved parameters that were carried into the new collection's config
 *     (after defaults and trial-axis mapping)
 * @param documentsCopied number of live documents copied from the old collection into the new one
 * @param rebuildMillis wall-time of the rebuild in milliseconds
 * @param persistenceRefreshed {@code true} iff the rebuild kept the on-disk image in sync with the
 *     in-memory state ({@code true} for in-memory collections; {@code false} for collections
 *     originally opened from a persistent directory)
 */
public record ApplyTrialResult(
    Map<String, Object> appliedParams,
    int documentsCopied,
    long rebuildMillis,
    boolean persistenceRefreshed) {

  public ApplyTrialResult {
    appliedParams = Map.copyOf(appliedParams);
  }
}
