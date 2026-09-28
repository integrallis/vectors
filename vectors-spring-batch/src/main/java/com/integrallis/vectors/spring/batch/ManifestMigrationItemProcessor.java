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
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import org.springframework.batch.item.ItemProcessor;

/**
 * Migrates one collection per item, or reports what it would do.
 *
 * <p><b>Dry run by default.</b> The processor is constructed read-only, so a job assembled without
 * thinking about it reports rather than writes. That is the opposite of convenient and deliberately
 * so: the destructive setting should be the one somebody typed.
 *
 * <p>Errors are allowed to propagate. Spring Batch already owns the vocabulary for what to do with
 * a failing item — {@code faultTolerant().skip(IOException.class)}, a {@code SkipPolicy}, a retry —
 * and swallowing an {@link IOException} here would take that choice away and report a clean job
 * over a collection that never migrated. A corrupt manifest in particular must stop the item rather
 * than be counted as done.
 */
public class ManifestMigrationItemProcessor implements ItemProcessor<Path, MigrationReport> {

  private final boolean apply;

  /** Creates a read-only processor that reports what migration would do. */
  public ManifestMigrationItemProcessor() {
    this(false);
  }

  /**
   * Creates a processor.
   *
   * @param apply true to rewrite manifests, false to report only
   */
  public ManifestMigrationItemProcessor(boolean apply) {
    this.apply = apply;
  }

  @Override
  public MigrationReport process(Path collectionRoot) throws IOException {
    List<ManifestMigration.Result> results =
        apply
            ? ManifestMigration.migrate(collectionRoot)
            : ManifestMigration.inspect(collectionRoot);
    return MigrationReport.of(collectionRoot, results, apply);
  }

  /**
   * Whether this processor writes.
   *
   * @return true if manifests are rewritten
   */
  public boolean applies() {
    return apply;
  }
}
