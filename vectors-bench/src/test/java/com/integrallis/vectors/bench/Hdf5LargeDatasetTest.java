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
package com.integrallis.vectors.bench;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.bench.dataset.DatasetRegistry;
import com.integrallis.vectors.bench.dataset.Hdf5Loader;
import java.nio.file.Path;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Reads an ANN-Benchmarks dataset whose payload exceeds {@link Integer#MAX_VALUE} bytes.
 *
 * <p>GIST1M's {@code /train} is 1,000,000 x 960 floats — 3.84 GB — and a whole-dataset HDF5 read
 * maps it into a single {@link java.nio.ByteBuffer}, which cannot exceed 2 GB. That failed with
 * "Size exceeds Integer.MAX_VALUE" from inside the HDF5 library and made GIST1M and deep-image-96
 * simply unreadable by this harness. {@code Hdf5Loader} now reads such datasets in row blocks.
 *
 * <p>Tagged {@code slow} and gated on the file being present: there is no way to exercise a 2 GB
 * boundary without 2 GB of data, and fabricating a file that large in a unit test would cost more
 * than it proves. {@code ./gradlew :vectors-bench:slowTest} runs it once the dataset is downloaded.
 */
@Tag("slow")
@EnabledIf("com.integrallis.vectors.bench.Hdf5LargeDatasetTest#gistAvailable")
@DisplayName("Hdf5Loader on a dataset larger than 2 GB")
class Hdf5LargeDatasetTest {

  private static final String GIST = "gist-960-euclidean";

  /** JUnit condition: the GIST1M HDF5 file is present. */
  static boolean gistAvailable() {
    return DatasetRegistry.isAnnBenchAvailable(GIST);
  }

  @Test
  @DisplayName("reads all 1,000,000 x 960 rows, and the last row is not left null")
  void readsPastTheTwoGigabyteBoundary() {
    Path path = DatasetRegistry.annBenchDataset(GIST);

    float[][] train = Hdf5Loader.readTrainVectors(path);

    assertThat(train).hasDimensions(1_000_000, 960);
    // Block assembly is where an off-by-one would hide: a wrong block count or a wrong final block
    // size leaves trailing rows null, which a dimensions check alone would not catch because the
    // outer array is allocated at full length up front.
    assertThat(train[0]).isNotNull();
    assertThat(train[train.length - 1]).isNotNull();
    assertThat(train[train.length / 2]).isNotNull();
    for (int i = 0; i < train.length; i += 10_000) {
      assertThat(train[i]).as("row %d was never filled", i).isNotNull().hasSize(960);
    }

    // GIST vectors are non-negative descriptors, so an all-zero row at a block boundary would mean
    // a
    // block was read but written into the wrong place.
    assertThat(sum(train[train.length - 1])).isGreaterThan(0f);
  }

  @Test
  @DisplayName("queries and ground truth still read whole, being far below the boundary")
  void smallDatasetsUnaffected() {
    Path path = DatasetRegistry.annBenchDataset(GIST);
    assertThat(Hdf5Loader.readTestVectors(path)).hasDimensions(1_000, 960);
    assertThat(Hdf5Loader.readNeighbors(path)).hasDimensions(1_000, 100);
  }

  private static float sum(float[] row) {
    float total = 0;
    for (float f : row) {
      total += f;
    }
    return total;
  }
}
