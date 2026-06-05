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
import static org.assertj.core.api.Assertions.assertThatIOException;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.quantization.ArrayVectorDataset;
import com.integrallis.vectors.quantization.BinaryMode;
import com.integrallis.vectors.quantization.BinaryQuantizedVectors;
import com.integrallis.vectors.quantization.BinaryQuantizer;
import com.integrallis.vectors.quantization.CompressedVectors;
import com.integrallis.vectors.quantization.NVQuantizedVectors;
import com.integrallis.vectors.quantization.NVQuantizer;
import com.integrallis.vectors.quantization.PQVectors;
import com.integrallis.vectors.quantization.ProductQuantizer;
import com.integrallis.vectors.quantization.RaBitQuantizedVectors;
import com.integrallis.vectors.quantization.RaBitQuantizer;
import com.integrallis.vectors.quantization.ScalarBits;
import com.integrallis.vectors.quantization.ScalarQuantizedVectors;
import com.integrallis.vectors.quantization.ScalarQuantizer;
import com.integrallis.vectors.quantization.ScoreFunction;
import com.integrallis.vectors.quantization.TurboQuantizedVectors;
import com.integrallis.vectors.quantization.TurboQuantizer;
import com.integrallis.vectors.quantization.VectorDataset;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class QuantizedVectorsCodecTest {

  private static final int DIM = 64;
  private static final int NUM_VECTORS = 200;
  private static final long SEED = 42L;

  private static float[][] randomVectors(int count, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] vectors = new float[count][dim];
    for (int i = 0; i < count; i++) {
      for (int d = 0; d < dim; d++) {
        vectors[i][d] = (float) rng.nextGaussian();
      }
    }
    return vectors;
  }

  private static VectorDataset dataset() {
    return new ArrayVectorDataset(randomVectors(NUM_VECTORS, DIM, SEED));
  }

  @Nested
  class SQ8RoundTrip {

    @Test
    void roundTripProducesIdenticalScores() throws IOException {
      VectorDataset data = dataset();
      ScalarQuantizer quantizer = ScalarQuantizer.train(data, ScalarBits.INT8);
      ScalarQuantizedVectors original = quantizer.encodeAll(data);

      byte[] encoded = QuantizedVectorsCodec.encode(original, quantizer, QuantizerKind.SQ8);
      CompressedVectors decoded = QuantizedVectorsCodec.decode(encoded);

      assertThat(decoded).isInstanceOf(ScalarQuantizedVectors.class);
      ScalarQuantizedVectors sq = (ScalarQuantizedVectors) decoded;
      assertThat(sq.size()).isEqualTo(original.size());
      assertThat(sq.dimension()).isEqualTo(DIM);

      // Verify scores match within tolerance
      float[] query = randomVectors(1, DIM, 99L)[0];
      ScoreFunction origScorer = original.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
      ScoreFunction decodedScorer = sq.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
      for (int i = 0; i < NUM_VECTORS; i++) {
        assertThat(decodedScorer.score(i))
            .isCloseTo(origScorer.score(i), org.assertj.core.api.Assertions.within(1e-5f));
      }
    }

    @Test
    void encodeTwiceProducesSameBytes() {
      VectorDataset data = dataset();
      ScalarQuantizer quantizer = ScalarQuantizer.train(data, ScalarBits.INT8);
      ScalarQuantizedVectors compressed = quantizer.encodeAll(data);

      byte[] first = QuantizedVectorsCodec.encode(compressed, quantizer, QuantizerKind.SQ8);
      byte[] second = QuantizedVectorsCodec.encode(compressed, quantizer, QuantizerKind.SQ8);
      assertThat(first).isEqualTo(second);
    }
  }

  @Nested
  class SQ4RoundTrip {

    @Test
    void roundTripProducesIdenticalScores() throws IOException {
      VectorDataset data = dataset();
      ScalarQuantizer quantizer = ScalarQuantizer.train(data, ScalarBits.INT4);
      ScalarQuantizedVectors original = quantizer.encodeAll(data);

      byte[] encoded = QuantizedVectorsCodec.encode(original, quantizer, QuantizerKind.SQ4);
      CompressedVectors decoded = QuantizedVectorsCodec.decode(encoded);

      assertThat(decoded).isInstanceOf(ScalarQuantizedVectors.class);
      ScalarQuantizedVectors sq = (ScalarQuantizedVectors) decoded;
      assertThat(sq.size()).isEqualTo(original.size());

      float[] query = randomVectors(1, DIM, 99L)[0];
      ScoreFunction origScorer = original.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      ScoreFunction decodedScorer = sq.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      for (int i = 0; i < NUM_VECTORS; i++) {
        assertThat(decodedScorer.score(i))
            .isCloseTo(origScorer.score(i), org.assertj.core.api.Assertions.within(1e-5f));
      }
    }
  }

  @Nested
  class PQRoundTrip {

    @Test
    void roundTripProducesIdenticalScores() throws IOException {
      VectorDataset data = dataset();
      ProductQuantizer quantizer = ProductQuantizer.train(data, 8, 256, true);
      PQVectors original = quantizer.encodeAll(data);

      byte[] encoded = QuantizedVectorsCodec.encode(original, quantizer, QuantizerKind.PQ);
      CompressedVectors decoded = QuantizedVectorsCodec.decode(encoded);

      assertThat(decoded).isInstanceOf(PQVectors.class);
      PQVectors pq = (PQVectors) decoded;
      assertThat(pq.size()).isEqualTo(original.size());
      assertThat(pq.dimension()).isEqualTo(DIM);

      float[] query = randomVectors(1, DIM, 99L)[0];
      ScoreFunction origScorer = original.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
      ScoreFunction decodedScorer = pq.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
      for (int i = 0; i < NUM_VECTORS; i++) {
        assertThat(decodedScorer.score(i))
            .isCloseTo(origScorer.score(i), org.assertj.core.api.Assertions.within(1e-5f));
      }
    }

    @Test
    void roundTripWithoutGlobalCentroid() throws IOException {
      VectorDataset data = dataset();
      ProductQuantizer quantizer = ProductQuantizer.train(data, 8, 256, false);
      PQVectors original = quantizer.encodeAll(data);

      byte[] encoded = QuantizedVectorsCodec.encode(original, quantizer, QuantizerKind.PQ);
      CompressedVectors decoded = QuantizedVectorsCodec.decode(encoded);

      assertThat(decoded).isInstanceOf(PQVectors.class);
      PQVectors pq = (PQVectors) decoded;
      assertThat(pq.size()).isEqualTo(original.size());

      float[] query = randomVectors(1, DIM, 99L)[0];
      ScoreFunction origScorer = original.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      ScoreFunction decodedScorer = pq.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      for (int i = 0; i < NUM_VECTORS; i++) {
        assertThat(decodedScorer.score(i))
            .isCloseTo(origScorer.score(i), org.assertj.core.api.Assertions.within(1e-5f));
      }
    }
  }

  @Nested
  class BQRoundTrip {

    @Test
    void signBitRoundTrip() throws IOException {
      VectorDataset data = dataset();
      BinaryQuantizer quantizer = BinaryQuantizer.train(data, BinaryMode.SIGN_BIT);
      BinaryQuantizedVectors original = quantizer.encodeAll(data);

      byte[] encoded = QuantizedVectorsCodec.encode(original, quantizer, QuantizerKind.BQ);
      CompressedVectors decoded = QuantizedVectorsCodec.decode(encoded);

      assertThat(decoded).isInstanceOf(BinaryQuantizedVectors.class);
      BinaryQuantizedVectors bq = (BinaryQuantizedVectors) decoded;
      assertThat(bq.size()).isEqualTo(original.size());

      float[] query = randomVectors(1, DIM, 99L)[0];
      ScoreFunction origScorer = original.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
      ScoreFunction decodedScorer = bq.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
      for (int i = 0; i < NUM_VECTORS; i++) {
        assertThat(decodedScorer.score(i))
            .isCloseTo(origScorer.score(i), org.assertj.core.api.Assertions.within(1e-5f));
      }
    }

    @Test
    void bbqRoundTrip() throws IOException {
      VectorDataset data = dataset();
      BinaryQuantizer quantizer = BinaryQuantizer.train(data, BinaryMode.BBQ);
      BinaryQuantizedVectors original = quantizer.encodeAll(data);

      byte[] encoded = QuantizedVectorsCodec.encode(original, quantizer, QuantizerKind.BQ);
      CompressedVectors decoded = QuantizedVectorsCodec.decode(encoded);

      assertThat(decoded).isInstanceOf(BinaryQuantizedVectors.class);
      BinaryQuantizedVectors bq = (BinaryQuantizedVectors) decoded;
      assertThat(bq.size()).isEqualTo(original.size());

      float[] query = randomVectors(1, DIM, 99L)[0];
      ScoreFunction origScorer = original.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      ScoreFunction decodedScorer = bq.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      for (int i = 0; i < NUM_VECTORS; i++) {
        assertThat(decodedScorer.score(i))
            .isCloseTo(origScorer.score(i), org.assertj.core.api.Assertions.within(1e-5f));
      }
    }
  }

  @Nested
  class RaBitQRoundTrip {

    @Test
    void roundTripProducesIdenticalScores() throws IOException {
      VectorDataset data = dataset();
      RaBitQuantizer quantizer = RaBitQuantizer.train(data, SEED);
      RaBitQuantizedVectors original = quantizer.encodeAll(data);

      byte[] encoded = QuantizedVectorsCodec.encode(original, quantizer, QuantizerKind.RABITQ);
      CompressedVectors decoded = QuantizedVectorsCodec.decode(encoded);

      assertThat(decoded).isInstanceOf(RaBitQuantizedVectors.class);
      RaBitQuantizedVectors rq = (RaBitQuantizedVectors) decoded;
      assertThat(rq.size()).isEqualTo(original.size());
      assertThat(rq.dimension()).isEqualTo(DIM);

      float[] query = randomVectors(1, DIM, 99L)[0];
      ScoreFunction origScorer = original.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      ScoreFunction decodedScorer = rq.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      for (int i = 0; i < NUM_VECTORS; i++) {
        assertThat(decodedScorer.score(i))
            .isCloseTo(origScorer.score(i), org.assertj.core.api.Assertions.within(1e-5f));
      }
    }

    @Test
    void encodeTwiceProducesSameBytes() {
      VectorDataset data = dataset();
      RaBitQuantizer quantizer = RaBitQuantizer.train(data, SEED);
      RaBitQuantizedVectors compressed = quantizer.encodeAll(data);

      byte[] first = QuantizedVectorsCodec.encode(compressed, quantizer, QuantizerKind.RABITQ);
      byte[] second = QuantizedVectorsCodec.encode(compressed, quantizer, QuantizerKind.RABITQ);
      assertThat(first).isEqualTo(second);
    }
  }

  @Nested
  class NVQRoundTrip {

    @Test
    void roundTripProducesIdenticalScores() throws IOException {
      VectorDataset data = dataset();
      NVQuantizer quantizer = NVQuantizer.train(data, 2);
      NVQuantizedVectors original = quantizer.encodeAll(data);

      byte[] encoded = QuantizedVectorsCodec.encode(original, quantizer, QuantizerKind.NVQ);
      CompressedVectors decoded = QuantizedVectorsCodec.decode(encoded);

      assertThat(decoded).isInstanceOf(NVQuantizedVectors.class);
      NVQuantizedVectors nvq = (NVQuantizedVectors) decoded;
      assertThat(nvq.size()).isEqualTo(original.size());
      assertThat(nvq.dimension()).isEqualTo(DIM);

      float[] query = randomVectors(1, DIM, 99L)[0];
      ScoreFunction origScorer = original.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
      ScoreFunction decodedScorer = nvq.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
      for (int i = 0; i < NUM_VECTORS; i++) {
        assertThat(decodedScorer.score(i))
            .isCloseTo(origScorer.score(i), org.assertj.core.api.Assertions.within(1e-5f));
      }
    }

    @Test
    void encodeTwiceProducesSameBytes() {
      VectorDataset data = dataset();
      NVQuantizer quantizer = NVQuantizer.train(data, 2);
      NVQuantizedVectors compressed = quantizer.encodeAll(data);

      byte[] first = QuantizedVectorsCodec.encode(compressed, quantizer, QuantizerKind.NVQ);
      byte[] second = QuantizedVectorsCodec.encode(compressed, quantizer, QuantizerKind.NVQ);
      assertThat(first).isEqualTo(second);
    }
  }

  @Nested
  class TurboQuantRoundTrip {

    @Test
    void roundTripProducesIdenticalScores() throws IOException {
      VectorDataset data = dataset();
      TurboQuantizer quantizer = TurboQuantizer.train(data, 8, SEED);
      TurboQuantizedVectors original = quantizer.encodeAll(data);

      byte[] encoded = QuantizedVectorsCodec.encode(original, quantizer, QuantizerKind.TURBOQUANT);
      CompressedVectors decoded = QuantizedVectorsCodec.decode(encoded);

      assertThat(decoded).isInstanceOf(TurboQuantizedVectors.class);
      TurboQuantizedVectors tq = (TurboQuantizedVectors) decoded;
      assertThat(tq.size()).isEqualTo(original.size());
      assertThat(tq.dimension()).isEqualTo(DIM);

      float[] query = randomVectors(1, DIM, 99L)[0];
      ScoreFunction origScorer = original.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      ScoreFunction decodedScorer = tq.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      for (int i = 0; i < NUM_VECTORS; i++) {
        assertThat(decodedScorer.score(i))
            .isCloseTo(origScorer.score(i), org.assertj.core.api.Assertions.within(1e-5f));
      }
    }

    @Test
    void fourBitRoundTripProducesIdenticalScores() throws IOException {
      // 4-bit exercises sub-byte index packing across byte boundaries.
      VectorDataset data = dataset();
      TurboQuantizer quantizer = TurboQuantizer.train(data, 4, SEED);
      TurboQuantizedVectors original = quantizer.encodeAll(data);

      byte[] encoded = QuantizedVectorsCodec.encode(original, quantizer, QuantizerKind.TURBOQUANT);
      CompressedVectors decoded = QuantizedVectorsCodec.decode(encoded);

      TurboQuantizedVectors tq = (TurboQuantizedVectors) decoded;
      float[] query = randomVectors(1, DIM, 99L)[0];
      ScoreFunction origScorer = original.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      ScoreFunction decodedScorer = tq.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      for (int i = 0; i < NUM_VECTORS; i++) {
        assertThat(decodedScorer.score(i))
            .isCloseTo(origScorer.score(i), org.assertj.core.api.Assertions.within(1e-5f));
      }
    }

    @Test
    void encodeTwiceProducesSameBytes() {
      VectorDataset data = dataset();
      TurboQuantizer quantizer = TurboQuantizer.train(data, 8, SEED);
      TurboQuantizedVectors compressed = quantizer.encodeAll(data);

      byte[] first = QuantizedVectorsCodec.encode(compressed, quantizer, QuantizerKind.TURBOQUANT);
      byte[] second = QuantizedVectorsCodec.encode(compressed, quantizer, QuantizerKind.TURBOQUANT);
      assertThat(first).isEqualTo(second);
    }
  }

  @Nested
  class Validation {

    @Test
    void truncatedBytesRejected() {
      byte[] tooShort = new byte[QuantizedVectorsCodec.COMMON_HEADER_SIZE - 1];
      assertThatIOException().isThrownBy(() -> QuantizedVectorsCodec.decode(tooShort));
    }

    @Test
    void wrongMagicRejected() {
      VectorDataset data = dataset();
      ScalarQuantizer quantizer = ScalarQuantizer.train(data, ScalarBits.INT8);
      ScalarQuantizedVectors compressed = quantizer.encodeAll(data);
      byte[] encoded = QuantizedVectorsCodec.encode(compressed, quantizer, QuantizerKind.SQ8);

      // Corrupt magic
      encoded[0] ^= (byte) 0xFF;
      assertThatIOException()
          .isThrownBy(() -> QuantizedVectorsCodec.decode(encoded))
          .withMessageContaining("magic");
    }

    @Test
    void wrongVersionRejected() {
      VectorDataset data = dataset();
      ScalarQuantizer quantizer = ScalarQuantizer.train(data, ScalarBits.INT8);
      ScalarQuantizedVectors compressed = quantizer.encodeAll(data);
      byte[] encoded = QuantizedVectorsCodec.encode(compressed, quantizer, QuantizerKind.SQ8);

      // Corrupt version (bytes 4-7)
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(4, 999);
      assertThatIOException()
          .isThrownBy(() -> QuantizedVectorsCodec.decode(encoded))
          .withMessageContaining("version");
    }

    @Test
    void noneQuantizerKindRejected() {
      VectorDataset data = dataset();
      ScalarQuantizer quantizer = ScalarQuantizer.train(data, ScalarBits.INT8);
      ScalarQuantizedVectors compressed = quantizer.encodeAll(data);
      byte[] encoded = QuantizedVectorsCodec.encode(compressed, quantizer, QuantizerKind.SQ8);

      // Set kind ordinal to NONE (0)
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(8, QuantizerKind.NONE.ordinal());
      assertThatIOException()
          .isThrownBy(() -> QuantizedVectorsCodec.decode(encoded))
          .withMessageContaining("NONE");
    }

    @Test
    void invalidKindOrdinalRejected() {
      VectorDataset data = dataset();
      ScalarQuantizer quantizer = ScalarQuantizer.train(data, ScalarBits.INT8);
      ScalarQuantizedVectors compressed = quantizer.encodeAll(data);
      byte[] encoded = QuantizedVectorsCodec.encode(compressed, quantizer, QuantizerKind.SQ8);

      // Set kind ordinal to out-of-range
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(8, 9999);
      assertThatIOException()
          .isThrownBy(() -> QuantizedVectorsCodec.decode(encoded))
          .withMessageContaining("invalid quantizerKind");
    }

    @Test
    void truncatedVectorDataRejected() {
      VectorDataset data = dataset();
      ScalarQuantizer quantizer = ScalarQuantizer.train(data, ScalarBits.INT8);
      ScalarQuantizedVectors compressed = quantizer.encodeAll(data);
      byte[] full = QuantizedVectorsCodec.encode(compressed, quantizer, QuantizerKind.SQ8);

      // Truncate after quantizer state but before all vector data
      byte[] truncated = new byte[QuantizedVectorsCodec.COMMON_HEADER_SIZE + 12 + 10];
      System.arraycopy(full, 0, truncated, 0, truncated.length);
      assertThatIOException()
          .isThrownBy(() -> QuantizedVectorsCodec.decode(truncated))
          .withMessageContaining("truncated");
    }

    @Test
    void negativeDimensionRejected() {
      VectorDataset data = dataset();
      ScalarQuantizer quantizer = ScalarQuantizer.train(data, ScalarBits.INT8);
      ScalarQuantizedVectors compressed = quantizer.encodeAll(data);
      byte[] encoded = QuantizedVectorsCodec.encode(compressed, quantizer, QuantizerKind.SQ8);

      // Set dimension to negative
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(12, -1);
      assertThatIOException()
          .isThrownBy(() -> QuantizedVectorsCodec.decode(encoded))
          .withMessageContaining("dimension");
    }
  }
}
