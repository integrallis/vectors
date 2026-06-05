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
package com.integrallis.vectors.optimizer.study;

import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.QuantizerParams;
import com.integrallis.vectors.db.VectorCollectionBuilder;
import com.integrallis.vectors.db.VectorCollectionConfig;

final class IndexMemoryEstimator {

  private static final int IVF_DEFAULT_HARMONY_KEY_DIMS = 0;
  private static final int RABIT_CORRECTION_FLOATS = 5;
  private static final int BBQ_CORRECTION_FLOATS = 3;
  private static final int NVQ_METADATA_FLOATS_PER_SUBVECTOR = 4;

  private IndexMemoryEstimator() {}

  static long estimateIndexPayloadBytes(VectorCollectionConfig config, int physicalSize) {
    if (physicalSize < 0) {
      throw new IllegalArgumentException("physicalSize must be non-negative: " + physicalSize);
    }
    int dimension = config.dimension();
    long bytes = retainedVectorPayloadBytes(config, physicalSize, dimension);
    return bytes + routingPayloadBytes(config, physicalSize, dimension);
  }

  private static long retainedVectorPayloadBytes(
      VectorCollectionConfig config, int physicalSize, int dimension) {
    long rawBytes = rawVectorBytes(physicalSize, dimension);
    if (!supportsCollectionQuantizer(config.indexType())
        || config.quantizerKind() == QuantizerKind.NONE
        || physicalSize == 0) {
      return rawBytes;
    }
    return rawBytes + quantizedVectorPayloadBytes(config, physicalSize, dimension);
  }

  private static boolean supportsCollectionQuantizer(IndexType indexType) {
    return switch (indexType) {
      case FLAT, HNSW, VAMANA -> true;
      case IVF_FLAT, IVF_PQ, CUVS_BRUTEFORCE, CUVS_CAGRA -> false;
    };
  }

  private static long quantizedVectorPayloadBytes(
      VectorCollectionConfig config, int physicalSize, int dimension) {
    return switch (config.quantizerKind()) {
      case NONE -> 0L;
      case SQ8 -> scalarQuantizedBytes(physicalSize, dimension, dimension);
      case SQ4 -> scalarQuantizedBytes(physicalSize, dimension, (dimension + 1) / 2);
      case PQ -> productQuantizedBytes(config.quantizerParams(), physicalSize, dimension);
      case BQ -> binaryQuantizedBytes(config.quantizerParams(), physicalSize, dimension);
      case RABITQ -> rabitQuantizedBytes(physicalSize, dimension);
      case NVQ -> nvqBytes(config.quantizerParams(), physicalSize, dimension);
      case TURBOQUANT -> turboQuantizedBytes(config.quantizerParams(), physicalSize, dimension);
      // fp16: two bytes per coordinate, no quantizer state.
      case FP16 -> (long) physicalSize * dimension * Short.BYTES;
    };
  }

  private static long scalarQuantizedBytes(
      int physicalSize, int dimension, int encodedBytesPerVector) {
    long vectors = (long) physicalSize * (encodedBytesPerVector + Float.BYTES);
    long quantizerState = Integer.BYTES + 5L * Float.BYTES;
    return vectors + quantizerState;
  }

  private static long productQuantizedBytes(
      QuantizerParams params, int physicalSize, int dimension) {
    QuantizerParams.PqParams pq =
        params instanceof QuantizerParams.PqParams p
            ? p
            : new QuantizerParams.PqParams(
                Math.max(1, dimension / 8), VectorCollectionBuilder.DEFAULT_PQ_CLUSTERS, true, 1);
    int subspaces = pq.numSubspaces();
    int clusters = pq.numClusters();
    long codes = (long) physicalSize * subspaces;
    long subspaceLayout = (long) subspaces * 2L * Integer.BYTES;
    long codebooks = (long) clusters * dimension * Float.BYTES;
    long centroid = pq.center() ? rawVectorBytes(1, dimension) : 0L;
    return codes + subspaceLayout + codebooks + centroid + Integer.BYTES * 3L;
  }

  private static long binaryQuantizedBytes(
      QuantizerParams params, int physicalSize, int dimension) {
    QuantizerParams.BqParams bq =
        params instanceof QuantizerParams.BqParams p ? p : new QuantizerParams.BqParams(true);
    long codeBytesPerVector = (long) ((dimension + 63) / 64) * Long.BYTES;
    long correctionBytes =
        bq.bbq() ? (long) physicalSize * BBQ_CORRECTION_FLOATS * Float.BYTES : 0L;
    long centroid = bq.bbq() ? rawVectorBytes(1, dimension) : 0L;
    return (long) physicalSize * codeBytesPerVector + correctionBytes + centroid;
  }

  private static long rabitQuantizedBytes(int physicalSize, int dimension) {
    int paddedDimension = ((dimension + 63) / 64) * 64;
    long codeBytesPerVector = (long) (paddedDimension / 64) * Long.BYTES;
    long corrections = (long) physicalSize * RABIT_CORRECTION_FLOATS * Float.BYTES;
    long centroids = rawVectorBytes(1, dimension) + rawVectorBytes(1, paddedDimension);
    long givensRotation = (long) (paddedDimension / 2) * 2L * Float.BYTES;
    return (long) physicalSize * codeBytesPerVector + corrections + centroids + givensRotation;
  }

  private static long turboQuantizedBytes(QuantizerParams params, int physicalSize, int dimension) {
    QuantizerParams.TurboParams tp = params instanceof QuantizerParams.TurboParams p ? p : null;
    int bits = tp != null ? tp.bits() : VectorCollectionBuilder.DEFAULT_TURBO_BITS;
    boolean unbiased = tp != null ? tp.unbiased() : VectorCollectionBuilder.DEFAULT_TURBO_UNBIASED;
    int paddedDimension = ((dimension + 63) / 64) * 64;
    // Per vector: packed indices ((paddedDim*bits+7)/8) + norm float; the unbiased path adds the
    // QJL
    // sign bits (paddedDim/8) + residual-norm float. Codebooks/sketch are not stored per vector.
    long indicesBytesPerVector = (long) (paddedDimension * bits + 7) / 8;
    long bytesPerVector = indicesBytesPerVector + Float.BYTES;
    if (unbiased) {
      bytesPerVector += (long) (paddedDimension + 7) / 8 + Float.BYTES;
    }
    long centroid = rawVectorBytes(1, dimension);
    // Paper-faithful default is a dense random rotation (~paddedDim^2 floats); the unbiased path
    // keeps an additional paddedDim×paddedDim QJL sketch resident.
    long rotationState = (long) paddedDimension * paddedDimension * Float.BYTES;
    long qjlSketch = unbiased ? (long) paddedDimension * paddedDimension * Float.BYTES : 0L;
    return (long) physicalSize * bytesPerVector + centroid + rotationState + qjlSketch;
  }

  private static long nvqBytes(QuantizerParams params, int physicalSize, int dimension) {
    QuantizerParams.NvqParams nvq =
        params instanceof QuantizerParams.NvqParams p
            ? p
            : new QuantizerParams.NvqParams(Math.max(1, dimension / 4));
    int subvectors = nvq.numSubvectors();
    long bytesPerVector =
        (long) dimension + (long) subvectors * NVQ_METADATA_FLOATS_PER_SUBVECTOR * Float.BYTES;
    long quantizerState = rawVectorBytes(1, dimension) + (long) subvectors * Integer.BYTES;
    return (long) physicalSize * bytesPerVector + quantizerState;
  }

  private static long routingPayloadBytes(
      VectorCollectionConfig config, int physicalSize, int dimension) {
    if (physicalSize == 0) {
      return 0L;
    }
    return switch (config.indexType()) {
      case FLAT, CUVS_BRUTEFORCE -> 0L;
      case HNSW -> hnswGraphBytes(config.hnswParams(), physicalSize);
      case VAMANA -> vamanaGraphBytes(config.vamanaParams(), physicalSize);
      case IVF_FLAT -> ivfRoutingBytes(config, physicalSize, dimension, false);
      case IVF_PQ -> ivfRoutingBytes(config, physicalSize, dimension, true);
      case CUVS_CAGRA -> cuvsCagraBytes(config.cuvsParams(), physicalSize);
    };
  }

  private static long hnswGraphBytes(VectorCollectionConfig.HnswParams params, int physicalSize) {
    int m = params == null ? VectorCollectionBuilder.DEFAULT_HNSW_M : params.m();
    long headerAndLevels = 32L + (long) physicalSize * Integer.BYTES;
    long layer0 = (long) physicalSize * (Integer.BYTES + (long) (2 * m) * Integer.BYTES);
    long expectedUpperNodeLayers =
        Math.max(1L, Math.ceilDiv((long) physicalSize, Math.max(1, m - 1)));
    long upper =
        expectedUpperNodeLayers * (Integer.BYTES + Integer.BYTES + (long) m * Integer.BYTES);
    return headerAndLevels + layer0 + upper;
  }

  private static long vamanaGraphBytes(
      VectorCollectionConfig.VamanaParams params, int physicalSize) {
    int maxDegree = params == null ? VectorCollectionBuilder.DEFAULT_VAMANA_R : params.maxDegree();
    return 28L + (long) physicalSize * (Integer.BYTES + (long) maxDegree * Integer.BYTES);
  }

  private static long ivfRoutingBytes(
      VectorCollectionConfig config, int physicalSize, int dimension, boolean pq) {
    int k;
    int pqSubspaces = 0;
    int pqClusters = 0;
    if (pq) {
      VectorCollectionConfig.IvfPqParams p = config.ivfPqParams();
      k = p == null ? VectorCollectionBuilder.DEFAULT_IVF_K : p.k();
      pqSubspaces = p == null ? Math.max(1, dimension / 8) : p.pqSubspaces();
      pqClusters = p == null ? VectorCollectionBuilder.DEFAULT_PQ_CLUSTERS : p.pqClusters();
    } else {
      VectorCollectionConfig.IvfParams p = config.ivfParams();
      k = p == null ? VectorCollectionBuilder.DEFAULT_IVF_K : p.k();
    }
    int effectiveK = Math.min(k, physicalSize);
    long buoy =
        Integer.BYTES
            + Integer.BYTES
            + Short.BYTES
            + config.metric().name().length()
            + (long) effectiveK * dimension * Float.BYTES
            + (long) effectiveK * Integer.BYTES
            + (long) effectiveK * Integer.BYTES
            + (long) effectiveK * Float.BYTES;
    long partitions =
        Integer.BYTES
            + buoy
            + Integer.BYTES
            + (long) effectiveK * 2L * Integer.BYTES
            + (long) physicalSize * Integer.BYTES
            + (long) effectiveK * IVF_DEFAULT_HARMONY_KEY_DIMS * Integer.BYTES
            + Byte.BYTES;
    if (!pq) {
      return partitions;
    }
    long pqTrailer =
        Integer.BYTES * 3L
            + Float.BYTES
            + Byte.BYTES
            + (long) pqSubspaces * 2L * Integer.BYTES
            + (long) pqClusters * dimension * Float.BYTES
            + Integer.BYTES
            + (long) physicalSize * pqSubspaces;
    return partitions + pqTrailer;
  }

  private static long cuvsCagraBytes(VectorCollectionConfig.CuVsParams params, int physicalSize) {
    int graphDegree =
        params instanceof VectorCollectionConfig.CuVsParams.Cagra cagra
            ? cagra.graphDegree()
            : VectorCollectionConfig.CuVsParams.Cagra.defaults().graphDegree();
    return (long) physicalSize * graphDegree * Integer.BYTES;
  }

  private static long rawVectorBytes(int physicalSize, int dimension) {
    return (long) physicalSize * dimension * Float.BYTES;
  }
}
