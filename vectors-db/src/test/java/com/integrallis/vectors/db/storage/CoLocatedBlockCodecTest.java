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
package com.integrallis.vectors.db.storage;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class CoLocatedBlockCodecTest {

  @Test
  void roundTripsAdjacencyAndCoLocatedCodes() {
    int numNodes = 500;
    int maxDegree = 32;
    int codeBytes = 50; // e.g. Ext-RaBitQ 4-bit over dim 100
    Random r = new Random(7);

    int[][] adj = new int[numNodes][];
    byte[][] codes = new byte[numNodes][codeBytes];
    for (int i = 0; i < numNodes; i++) {
      r.nextBytes(codes[i]);
    }
    for (int i = 0; i < numNodes; i++) {
      int deg = r.nextInt(maxDegree + 1); // 0..maxDegree
      adj[i] = new int[deg];
      for (int j = 0; j < deg; j++) {
        adj[i][j] = r.nextInt(numNodes);
      }
    }
    int entryNode = 42;

    byte[] blob = CoLocatedBlockCodec.encode(numNodes, maxDegree, codeBytes, entryNode, adj, codes);
    // exact fixed-stride size
    assertThat(blob.length)
        .isEqualTo(
            CoLocatedBlockCodec.HEADER_SIZE
                + numNodes * CoLocatedBlockCodec.strideFor(maxDegree, codeBytes));

    CoLocatedBlockCodec.Reader reader = CoLocatedBlockCodec.open(MemorySegment.ofArray(blob));
    assertThat(reader.numNodes()).isEqualTo(numNodes);
    assertThat(reader.maxDegree()).isEqualTo(maxDegree);
    assertThat(reader.codeBytes()).isEqualTo(codeBytes);
    assertThat(reader.entryNode()).isEqualTo(entryNode);

    int[] out = new int[maxDegree];
    for (int i = 0; i < numNodes; i++) {
      int deg = reader.neighbors(i, out);
      assertThat(deg).as("degree[%d]", i).isEqualTo(adj[i].length);
      for (int j = 0; j < deg; j++) {
        assertThat(out[j]).as("neighbor[%d][%d]", i, j).isEqualTo(adj[i][j]);
      }
      // node's own co-located code
      assertThat(sliceToBytes(reader.nodeCode(i), codeBytes))
          .as("nodeCode[%d]", i)
          .isEqualTo(codes[i]);
      // each neighbor's co-located code == that neighbor node's code
      for (int j = 0; j < deg; j++) {
        assertThat(sliceToBytes(reader.neighborCode(i, j), codeBytes))
            .as("neighborCode[%d][%d]", i, j)
            .isEqualTo(codes[adj[i][j]]);
      }
    }
  }

  @Test
  void blockOffsetIsFixedStrideArithmetic() {
    int maxDegree = 16, codeBytes = 8;
    byte[][] codes = new byte[10][codeBytes];
    int[][] adj = new int[10][0];
    byte[] blob = CoLocatedBlockCodec.encode(10, maxDegree, codeBytes, -1, adj, codes);
    var reader = CoLocatedBlockCodec.open(MemorySegment.ofArray(blob));
    int stride = CoLocatedBlockCodec.strideFor(maxDegree, codeBytes);
    for (int n = 0; n < 10; n++) {
      assertThat(reader.blockOffset(n))
          .isEqualTo((long) CoLocatedBlockCodec.HEADER_SIZE + (long) n * stride);
    }
  }

  @Test
  void rejectsBadMagic() {
    byte[] junk = new byte[64];
    assertThatThrownBy(() -> CoLocatedBlockCodec.open(MemorySegment.ofArray(junk)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("magic");
  }

  private static byte[] sliceToBytes(MemorySegment slice, int n) {
    byte[] b = new byte[n];
    MemorySegment.copy(slice, ValueLayout.JAVA_BYTE, 0, b, 0, n);
    return b;
  }
}
