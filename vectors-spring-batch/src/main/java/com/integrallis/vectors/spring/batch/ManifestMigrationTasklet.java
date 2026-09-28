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

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.springframework.batch.core.StepContribution;
import org.springframework.batch.core.scope.context.ChunkContext;
import org.springframework.batch.core.step.tasklet.Tasklet;
import org.springframework.batch.repeat.RepeatStatus;

/**
 * One-step migration of every collection under a directory.
 *
 * <p>The simple shape, for when a job has one thing to do:
 *
 * <pre>{@code
 * @Bean
 * Step migrateVectors(JobRepository repository, PlatformTransactionManager tx) {
 *   return new StepBuilder("migrateVectors", repository)
 *       .tasklet(new ManifestMigrationTasklet(Path.of("/var/lib/vectors"), true), tx)
 *       .build();
 * }
 * }</pre>
 *
 * <p>For many collections, where per-item skip, retry and restart matter, use {@link
 * CollectionRootItemReader} with {@link ManifestMigrationItemProcessor} in a chunk-oriented step
 * instead. This tasklet migrates everything inside one transaction boundary and reports totals; it
 * does not give Spring Batch a chance to skip one bad collection and carry on.
 *
 * <p>Counts land on the step's {@code writeCount} so a job's own metrics say how many generations
 * were rewritten, and the per-collection reports are put in the step execution context under {@link
 * #REPORTS_KEY} for a listener to log or assert on.
 */
public class ManifestMigrationTasklet implements Tasklet {

  /** Step execution context key holding {@code List<MigrationReport>}. */
  public static final String REPORTS_KEY = "vectors.migration.reports";

  private final Path base;
  private final boolean apply;

  /** Creates a read-only tasklet that reports what migration would do. */
  public ManifestMigrationTasklet(Path base) {
    this(base, false);
  }

  /**
   * Creates a tasklet.
   *
   * @param base a collection root, or a directory containing collection roots
   * @param apply true to rewrite manifests, false to report only
   */
  public ManifestMigrationTasklet(Path base, boolean apply) {
    this.base = Objects.requireNonNull(base, "base");
    this.apply = apply;
  }

  @Override
  public RepeatStatus execute(StepContribution contribution, ChunkContext chunkContext)
      throws Exception {
    CollectionRootItemReader reader = new CollectionRootItemReader(base);
    ManifestMigrationItemProcessor processor = new ManifestMigrationItemProcessor(apply);

    List<MigrationReport> reports = new ArrayList<>();
    int generations = 0;
    for (Path root = reader.read(); root != null; root = reader.read()) {
      // ItemProcessor.process is nullable by contract: returning null filters the item. This
      // implementation never does, but going through the interface is what makes that a contract
      // rather than an assumption, so the null is handled instead of trusted.
      MigrationReport report = processor.process(root);
      if (report == null) {
        continue;
      }
      reports.add(report);
      generations += report.migrated();
    }

    contribution.incrementWriteCount(generations);
    chunkContext
        .getStepContext()
        .getStepExecution()
        .getExecutionContext()
        .put(REPORTS_KEY, reports);
    return RepeatStatus.FINISHED;
  }
}
