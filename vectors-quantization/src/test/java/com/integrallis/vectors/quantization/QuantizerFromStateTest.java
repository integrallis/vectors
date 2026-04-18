package com.integrallis.vectors.quantization;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Round-trip tests for quantizer and rotation {@code fromState()} factory methods. Each test trains
 * a quantizer, extracts its internal state via accessors, reconstructs it via {@code fromState()},
 * and verifies that the reconstructed quantizer produces identical scores.
 */
class QuantizerFromStateTest {

  private static final int DIM = 32;
  private static final int NUM_VECTORS = 100;
  private static final long SEED = 42L;

  private static float[][] randomVectors(int count, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] vecs = new float[count][dim];
    for (int i = 0; i < count; i++) {
      for (int d = 0; d < dim; d++) {
        vecs[i][d] = (float) rng.nextGaussian();
      }
    }
    return vecs;
  }

  private static VectorDataset dataset() {
    return new ArrayVectorDataset(randomVectors(NUM_VECTORS, DIM, SEED));
  }

  @Nested
  @Tag("unit")
  class RotationFromState {

    @Test
    void randomRotation_fromMatrix_roundTrip() {
      RandomRotation original = RandomRotation.generate(DIM, SEED);
      RandomRotation restored =
          RandomRotation.fromMatrix(DIM, original.matrix(), original.matrixT());

      float[] input = randomVectors(1, DIM, 99L)[0];
      float[] rotatedOriginal = original.rotate(input);
      float[] rotatedRestored = restored.rotate(input);

      assertThat(rotatedRestored).containsExactly(rotatedOriginal);
    }

    @Test
    void givensRotation_fromCosSin_roundTrip() {
      GivensRotation original = GivensRotation.generate(DIM, SEED);
      GivensRotation restored = GivensRotation.fromCosSin(DIM, original.cos(), original.sin());

      float[] input = randomVectors(1, DIM, 99L)[0];
      float[] rotatedOriginal = original.rotate(input);
      float[] rotatedRestored = restored.rotate(input);

      assertThat(rotatedRestored).containsExactly(rotatedOriginal);
    }

    @Test
    void quaternionRotation_fromQuaternions_roundTrip() {
      QuaternionRotation original = QuaternionRotation.generate(DIM, SEED);
      QuaternionRotation restored =
          QuaternionRotation.fromQuaternions(DIM, original.qL(), original.qR());

      float[] input = randomVectors(1, DIM, 99L)[0];
      float[] rotatedOriginal = original.rotate(input);
      float[] rotatedRestored = restored.rotate(input);

      assertThat(rotatedRestored).containsExactly(rotatedOriginal);
    }
  }

  @Nested
  @Tag("unit")
  class ProductQuantizerFromState {

    @Test
    void roundTrip_producesIdenticalScores() {
      VectorDataset ds = dataset();
      ProductQuantizer original = ProductQuantizer.train(ds, 4, 16, true);
      PQVectors originalCompressed = original.encodeAll(ds);

      ProductQuantizer restored =
          ProductQuantizer.fromState(
              original.dimension(),
              original.numSubspaces(),
              original.numClusters(),
              original.subspaceSizesAndOffsets(),
              original.codebooks(),
              original.globalCentroid());
      PQVectors restoredCompressed = restored.encodeAll(ds);

      float[] query = randomVectors(1, DIM, 77L)[0];
      ScoreFunction sfOrig =
          originalCompressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      ScoreFunction sfRest =
          restoredCompressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);

      for (int i = 0; i < NUM_VECTORS; i++) {
        assertThat(sfRest.score(i)).isEqualTo(sfOrig.score(i));
      }
    }

    @Test
    void roundTrip_encodesIdenticalBytes() {
      VectorDataset ds = dataset();
      ProductQuantizer original = ProductQuantizer.train(ds, 4, 16, true);

      ProductQuantizer restored =
          ProductQuantizer.fromState(
              original.dimension(),
              original.numSubspaces(),
              original.numClusters(),
              original.subspaceSizesAndOffsets(),
              original.codebooks(),
              original.globalCentroid());

      float[] vec = ds.getVector(0);
      assertThat(restored.encode(vec)).isEqualTo(original.encode(vec));
    }
  }

  @Nested
  @Tag("unit")
  class BinaryQuantizerFromState {

    @Test
    void roundTrip_signBit_producesIdenticalScores() {
      VectorDataset ds = dataset();
      BinaryQuantizer original = BinaryQuantizer.train(ds, BinaryMode.SIGN_BIT);
      BinaryQuantizedVectors originalCompressed = original.encodeAll(ds);

      BinaryQuantizer restored =
          BinaryQuantizer.fromState(original.dimension(), original.mode(), original.centroid());
      BinaryQuantizedVectors restoredCompressed = restored.encodeAll(ds);

      float[] query = randomVectors(1, DIM, 77L)[0];
      ScoreFunction sfOrig =
          originalCompressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      ScoreFunction sfRest =
          restoredCompressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);

      for (int i = 0; i < NUM_VECTORS; i++) {
        assertThat(sfRest.score(i)).isEqualTo(sfOrig.score(i));
      }
    }

    @Test
    void roundTrip_bbq_producesIdenticalScores() {
      VectorDataset ds = dataset();
      BinaryQuantizer original = BinaryQuantizer.train(ds, BinaryMode.BBQ);
      BinaryQuantizedVectors originalCompressed = original.encodeAll(ds);

      BinaryQuantizer restored =
          BinaryQuantizer.fromState(original.dimension(), original.mode(), original.centroid());
      BinaryQuantizedVectors restoredCompressed = restored.encodeAll(ds);

      float[] query = randomVectors(1, DIM, 77L)[0];
      ScoreFunction sfOrig =
          originalCompressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      ScoreFunction sfRest =
          restoredCompressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);

      for (int i = 0; i < NUM_VECTORS; i++) {
        assertThat(sfRest.score(i)).isEqualTo(sfOrig.score(i));
      }
    }
  }

  @Nested
  @Tag("unit")
  class RaBitQuantizerFromState {

    @Test
    void roundTrip_producesIdenticalScores() {
      VectorDataset ds = dataset();
      RaBitQuantizer original = RaBitQuantizer.train(ds, SEED);
      RaBitQuantizedVectors originalCompressed = original.encodeAll(ds);

      RaBitQuantizer restored =
          RaBitQuantizer.fromState(
              original.dimension(),
              original.paddedDimension(),
              original.centroid(),
              original.rotation());

      RaBitQuantizedVectors restoredCompressed = restored.encodeAll(ds);

      float[] query = randomVectors(1, DIM, 77L)[0];
      ScoreFunction sfOrig =
          originalCompressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      ScoreFunction sfRest =
          restoredCompressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);

      // RaBitQ scores involve dot-product accumulators over padded dimensions; floating-point
      // accumulation order can differ by a small epsilon between two identical-state quantizers.
      for (int i = 0; i < NUM_VECTORS; i++) {
        assertThat(sfRest.score(i))
            .isCloseTo(sfOrig.score(i), org.assertj.core.data.Offset.offset(1e-5f));
      }
    }
  }

  @Nested
  @Tag("unit")
  class NVQuantizerFromState {

    @Test
    void roundTrip_producesIdenticalScores() {
      VectorDataset ds = dataset();
      NVQuantizer original = NVQuantizer.train(ds, 2);
      NVQuantizedVectors originalCompressed = original.encodeAll(ds);

      NVQuantizer restored =
          NVQuantizer.fromState(
              original.dimension(),
              original.numSubvectors(),
              original.subvectorSizes(),
              original.globalMean());

      NVQuantizedVectors restoredCompressed = restored.encodeAll(ds);

      float[] query = randomVectors(1, DIM, 77L)[0];
      ScoreFunction sfOrig =
          originalCompressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      ScoreFunction sfRest =
          restoredCompressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);

      for (int i = 0; i < NUM_VECTORS; i++) {
        assertThat(sfRest.score(i)).isEqualTo(sfOrig.score(i));
      }
    }

    @Test
    void roundTrip_encodesIdenticalBytes() {
      VectorDataset ds = dataset();
      NVQuantizer original = NVQuantizer.train(ds, 2);

      NVQuantizer restored =
          NVQuantizer.fromState(
              original.dimension(),
              original.numSubvectors(),
              original.subvectorSizes(),
              original.globalMean());

      float[] vec = ds.getVector(0);
      assertThat(restored.encode(vec)).isEqualTo(original.encode(vec));
    }
  }
}
