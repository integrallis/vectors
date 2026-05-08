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
package com.integrallis.vectors.studio.distributed;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.ivf.TierPolicy;
import com.integrallis.vectors.studio.core.connection.ConnectionConfig;
import java.nio.file.Path;

/**
 * Connection settings for a Cloudflare R2-backed {@link
 * com.integrallis.vectors.ivf.DistributedVectorCollection}. Used by {@link
 * DistributedStudioBackendProvider} to construct a {@link DistributedStudioBackend}.
 */
public record DistributedConnectionConfig(
    String collectionName,
    String s3Endpoint,
    String s3Region,
    String s3Bucket,
    String s3Prefix,
    String s3AccessKey,
    String s3SecretKey,
    Path walDir,
    int dim,
    SimilarityFunction metric,
    TierPolicy tierPolicy)
    implements ConnectionConfig {}
