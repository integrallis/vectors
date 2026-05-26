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
package com.integrallis.vectors.distributed;

/**
 * Metadata sent with node-to-node calls.
 *
 * @param bearerToken shared cluster token for authenticating internal node calls; {@code null}
 *     means no token is configured
 */
public record NodeCallContext(String bearerToken) {

  private static final NodeCallContext NONE = new NodeCallContext(null);

  /** Returns an unauthenticated context. */
  public static NodeCallContext none() {
    return NONE;
  }

  /** Returns a context carrying the configured cluster bearer token. */
  public static NodeCallContext bearer(String token) {
    return new NodeCallContext(token);
  }
}
