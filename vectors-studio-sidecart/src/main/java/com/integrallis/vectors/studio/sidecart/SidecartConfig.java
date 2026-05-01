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

import com.integrallis.vectors.studio.sidecart.sources.FileSidecartSource;
import com.integrallis.vectors.studio.sidecart.sources.H2SidecartSource;
import com.integrallis.vectors.studio.sidecart.sources.HttpSidecartSource;
import java.nio.file.Path;
import java.util.Map;

/**
 * Sealed configuration carrier for a single sidecart source. {@link #build()} materialises the
 * concrete {@link SidecartSource}; configuration loaders (YAML / properties / programmatic) deal
 * exclusively in instances of this type so that the SPI surface stays small.
 */
public sealed interface SidecartConfig {

  /** Materialises the configured source. */
  SidecartSource build();

  /** File-backed source, used for image directories or text-per-file dumps. */
  record File(Path baseDir, String extension, boolean textMode, String mimeType)
      implements SidecartConfig {
    @Override
    public SidecartSource build() {
      return new FileSidecartSource(baseDir, extension, textMode, mimeType);
    }
  }

  /** H2 / generic-JDBC source. {@code password} may be empty. */
  record H2(
      String jdbcUrl,
      String user,
      String password,
      String table,
      String idColumn,
      String textColumn,
      String blobColumn,
      String mimeColumn)
      implements SidecartConfig {
    @Override
    public SidecartSource build() {
      return new H2SidecartSource(
          jdbcUrl, user, password, table, idColumn, textColumn, blobColumn, mimeColumn);
    }
  }

  /** Generic HTTP source, used for presigned-URL access to S3 / GCS / R2 / file servers. */
  record Http(String urlTemplate, boolean textMode, String defaultMime, Map<String, String> headers)
      implements SidecartConfig {
    @Override
    public SidecartSource build() {
      return new HttpSidecartSource(urlTemplate, textMode, defaultMime, headers);
    }
  }
}
