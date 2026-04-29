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
package com.integrallis.vectors.studio.web;

import com.integrallis.vectors.studio.core.StudioSession;
import com.integrallis.vectors.studio.web.projection.ProjectionJobManager;
import io.helidon.webserver.WebServer;

/** Handle for a running Studio server. Caller closes the handle to stop the server. */
public final class StudioServerHandle implements AutoCloseable {

  private final WebServer server;
  private final StudioSession session;
  private final ProjectionJobManager jobs;

  StudioServerHandle(WebServer server, StudioSession session, ProjectionJobManager jobs) {
    this.server = server;
    this.session = session;
    this.jobs = jobs;
  }

  /** The listen port (resolved after {@link WebServer#start() start} for ephemeral-port setups). */
  public int port() {
    return server.port();
  }

  /** Stops the server and releases all backend / job-manager resources. */
  public void stop() {
    try {
      server.stop();
    } finally {
      try {
        jobs.close();
      } finally {
        session.close();
      }
    }
  }

  @Override
  public void close() {
    stop();
  }
}
