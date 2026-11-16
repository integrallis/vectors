/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
package com.integrallis.vectors.server.routing;

/**
 * Lets the process embedding {@code vectors-server} add its own lines to {@code GET /v1/metrics}.
 * Each contributor appends Prometheus text-exposition lines (its own {@code # HELP} / {@code #
 * TYPE} headers and samples, each line newline-terminated) after the engine's metrics. A
 * contributor that throws is skipped for that scrape; the engine's metrics are still served.
 */
@FunctionalInterface
public interface MetricsContributor {

  /**
   * @param out the exposition text under construction; append complete lines only
   */
  void appendTo(StringBuilder out);
}
