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
package com.integrallis.vectors.cluster;

import com.integrallis.vectors.db.VectorCollection;
import java.nio.file.Path;

/**
 * Creates the {@link VectorCollection} backing a single shard of a {@link ClusterVectorCollection}.
 *
 * <p>The factory is invoked once per shard at cluster-build time. Each shard is a fully independent
 * {@code vectors-db} collection — there is no shared index and no coordination between shards.
 */
@FunctionalInterface
public interface ShardFactory {

  /**
   * Builds the collection for one shard.
   *
   * @param shardIndex the shard ordinal in {@code [0, shardCount)}
   * @param shardStorageRoot the absolute per-shard storage directory ({@code
   *     <clusterRoot>/shard-<index>}), or {@code null} when the cluster was configured without a
   *     storage root (in-memory shards). Implementations that build persistent collections must
   *     pass this to {@link com.integrallis.vectors.db.VectorCollectionBuilder#storagePath(Path)}.
   * @return the shard's collection; must not be null
   */
  VectorCollection create(int shardIndex, Path shardStorageRoot);
}
