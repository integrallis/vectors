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
package com.integrallis.vectors.spring.batch;

import com.integrallis.vectors.db.storage.ManifestMigration;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * What happened to one collection, as one batch item.
 *
 * @param collectionRoot the collection
 * @param applied true if manifests were rewritten, false for a dry run
 * @param migrated generations migrated, or that would be
 * @param alreadyCurrent generations already at the current format
 * @param newerThanThisBuild generations written by a newer build and left alone
 * @param unsupported generations this build cannot read at all
 */
public record MigrationReport(
    Path collectionRoot,
    boolean applied,
    int migrated,
    int alreadyCurrent,
    int newerThanThisBuild,
    int unsupported) {

  /** Compact constructor. */
  public MigrationReport {
    Objects.requireNonNull(collectionRoot, "collectionRoot");
  }

  /**
   * Summarises one collection's results.
   *
   * @param collectionRoot the collection
   * @param results per-generation results
   * @param applied whether the manifests were actually rewritten
   * @return the report
   */
  public static MigrationReport of(
      Path collectionRoot, List<ManifestMigration.Result> results, boolean applied) {
    int migrated = 0;
    int current = 0;
    int newer = 0;
    int unsupported = 0;
    for (ManifestMigration.Result result : results) {
      switch (result.outcome()) {
        // Both mean "this generation is work": `applied` already says whether it was done.
        case MIGRATED, NEEDS_MIGRATION -> migrated++;
        case ALREADY_CURRENT -> current++;
        case NEWER_THAN_THIS_BUILD -> newer++;
        case UNSUPPORTED -> unsupported++;
      }
    }
    return new MigrationReport(collectionRoot, applied, migrated, current, newer, unsupported);
  }

  /**
   * Whether this collection needed no attention.
   *
   * @return true when every generation was already current
   */
  public boolean clean() {
    return migrated == 0 && newerThanThisBuild == 0 && unsupported == 0;
  }

  /**
   * Whether this collection cannot be opened by this build even after migrating.
   *
   * <p>Separated from {@link #clean()} because it is the one outcome a migration job cannot fix:
   * the answer is a newer library, not another run.
   *
   * @return true when a generation is newer than this build or unreadable
   */
  public boolean needsAttention() {
    return newerThanThisBuild > 0 || unsupported > 0;
  }

  @Override
  public String toString() {
    return (applied ? "migrated " : "would migrate ")
        + migrated
        + " generation(s) at "
        + collectionRoot
        + (alreadyCurrent > 0 ? ", " + alreadyCurrent + " already current" : "")
        + (newerThanThisBuild > 0 ? ", " + newerThanThisBuild + " NEWER than this build" : "")
        + (unsupported > 0 ? ", " + unsupported + " unreadable" : "");
  }
}
