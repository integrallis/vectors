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
package com.integrallis.vectors.jakarta.batch;

import com.integrallis.vectors.db.storage.ManifestMigration;
import jakarta.batch.api.AbstractBatchlet;
import jakarta.batch.api.BatchProperty;
import jakarta.inject.Inject;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Migrates every collection under a directory, as a JSR-352 batchlet.
 *
 * <p>The portable half of the migration adapters: this runs on any Jakarta Batch container — JBeret
 * under WildFly or Quarkus, Open Liberty's batch container — without Spring. Declared in a job's
 * JSL:
 *
 * <pre>{@code
 * <step id="migrate-vectors">
 *   <batchlet ref="manifestMigrationBatchlet">
 *     <properties>
 *       <property name="basePath" value="/var/lib/vectors"/>
 *       <property name="apply"    value="true"/>
 *     </properties>
 *   </batchlet>
 * </step>
 * }</pre>
 *
 * <p><b>Reports unless told otherwise.</b> {@code apply} defaults to false, so a job that forgets
 * the property describes what would change rather than changing it. Batch properties arrive as
 * strings and an absent one arrives as null, which is precisely the case that would otherwise
 * default to "write".
 *
 * <p>The exit status distinguishes four outcomes, so a JSL can branch on them: {@code MIGRATED}
 * when manifests were rewritten, {@code PENDING} when a report found work and wrote nothing, {@code
 * CLEAN} when there was nothing to do, and {@code ATTENTION} when some generation is newer than
 * this build or unreadable — a state another run will never fix. {@code MIGRATED} and {@code
 * PENDING} are kept apart deliberately: a dry run that reported MIGRATED is how an operator comes
 * to believe a migration ran.
 *
 * <p>Failures are not caught. A container's retry, restart and failure semantics are the reason to
 * be running inside one, and swallowing an {@link IOException} would report a step that succeeded
 * over a collection that never migrated.
 */
public class ManifestMigrationBatchlet extends AbstractBatchlet {

  /** Exit status when at least one generation was actually rewritten. */
  public static final String STATUS_MIGRATED = "MIGRATED";

  /**
   * Exit status when a report found work but wrote nothing.
   *
   * <p>Distinct from {@link #STATUS_MIGRATED} because conflating them is a trap: a job branching on
   * MIGRATED after a dry run would conclude the collections had been upgraded when nothing was
   * touched. A dry run that found work is a reason to schedule the real one, not a success.
   */
  public static final String STATUS_PENDING = "PENDING";

  /** Exit status when every generation was already current. */
  public static final String STATUS_CLEAN = "CLEAN";

  /** Exit status when something is newer than this build, or unreadable. */
  public static final String STATUS_ATTENTION = "ATTENTION";

  /** A collection root, or a directory containing collection roots. */
  @Inject
  @BatchProperty(name = "basePath")
  String basePath;

  /** {@code "true"} to rewrite manifests. Anything else, including absent, reports only. */
  @Inject
  @BatchProperty(name = "apply")
  String apply;

  /** Required by the container, which instantiates batch artifacts reflectively. */
  public ManifestMigrationBatchlet() {}

  /**
   * Constructor for programmatic use and tests, bypassing property injection.
   *
   * @param basePath a collection root, or a directory of collection roots
   * @param apply true to rewrite manifests
   */
  public ManifestMigrationBatchlet(Path basePath, boolean apply) {
    this.basePath = basePath == null ? null : basePath.toString();
    this.apply = Boolean.toString(apply);
  }

  /**
   * Migrates or reports, and returns the exit status the job branches on.
   *
   * @return {@link #STATUS_MIGRATED}, {@link #STATUS_PENDING}, {@link #STATUS_CLEAN} or {@link
   *     #STATUS_ATTENTION}
   * @throws IllegalStateException if {@code basePath} was not supplied
   * @throws IllegalArgumentException if {@code apply} is neither a boolean spelling nor blank
   * @throws java.io.IOException if a collection cannot be read or rewritten
   */
  @Override
  public String process() throws Exception {
    if (basePath == null || basePath.isBlank()) {
      throw new IllegalStateException(
          "the basePath batch property is required: it names the collection root, or a directory of"
              + " collection roots, to migrate");
    }
    boolean write = parseApply(apply);

    int migrated = 0;
    int attention = 0;
    for (Path root : ManifestMigration.collectionRoots(Path.of(basePath))) {
      List<ManifestMigration.Result> results =
          write ? ManifestMigration.migrate(root) : ManifestMigration.inspect(root);
      for (ManifestMigration.Result result : results) {
        switch (result.outcome()) {
          case MIGRATED, NEEDS_MIGRATION -> migrated++;
          case NEWER_THAN_THIS_BUILD, UNSUPPORTED -> attention++;
          case ALREADY_CURRENT -> {
            // Nothing to report.
          }
        }
      }
    }

    if (attention > 0) {
      return STATUS_ATTENTION;
    }
    if (migrated == 0) {
      return STATUS_CLEAN;
    }
    // Counted the same either way; what differs is whether it was written, and the status has to
    // say so. A dry run reporting MIGRATED is how an operator comes to believe a migration ran.
    return write ? STATUS_MIGRATED : STATUS_PENDING;
  }

  /**
   * Reads the {@code apply} batch property.
   *
   * <p>Absent or blank is false: a job that did not ask to write has not asked to write. Anything
   * that is not a boolean spelling is <b>refused</b> rather than treated as false, because silently
   * dry-running an operator who typed {@code apply="yes"} leaves them believing the collections
   * were upgraded. Case and surrounding whitespace are accepted, since neither signals uncertainty
   * of intent in hand-written XML.
   *
   * @param value the raw property value, possibly null
   * @return whether to rewrite manifests
   * @throws IllegalArgumentException if the value is neither a boolean spelling nor blank
   */
  private static boolean parseApply(String value) {
    if (value == null || value.isBlank()) {
      return false;
    }
    String normalised = value.trim();
    if ("true".equalsIgnoreCase(normalised)) {
      return true;
    }
    if ("false".equalsIgnoreCase(normalised)) {
      return false;
    }
    throw new IllegalArgumentException(
        "the apply batch property must be \"true\" or \"false\" (or absent, meaning false), but was \""
            + value
            + "\"; it is refused rather than read as false so a typo cannot become a dry run that"
            + " looks like a migration");
  }
}
