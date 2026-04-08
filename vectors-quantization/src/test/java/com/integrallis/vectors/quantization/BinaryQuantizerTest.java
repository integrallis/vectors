package com.integrallis.vectors.quantization;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.vectors.core.SimilarityFunction;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** Tests for {@link BinaryQuantizer} and {@link BinaryQuantizedVectors}. */
class BinaryQuantizerTest {

  /** Path to SIFT Small dataset (from JVector research repo). */
  private static final Path SIFT_DIR =
      Path.of("../../research/repos/jvector/siftsmall").toAbsolutePath().normalize();

  static boolean siftAvailable() {
    return Files.exists(SIFT_DIR.resolve("siftsmall_base.fvecs"));
  }

  @Nested
  @Tag("unit")
  class SignBitEncoding {

    @Test
    void signBitEncoding_positivesBecome1_negativesBecome0() {
      // 8-dim vector: alternating positive/negative
      float[] vector = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
      var dataset = new ArrayVectorDataset(new float[][] {vector});
      var bq = BinaryQuantizer.train(dataset, BinaryMode.SIGN_BIT);

      byte[] encoded = bq.encode(vector);
      // Bits: d0=1, d1=0, d2=1, d3=0, d4=1, d5=0, d6=1, d7=0
      // Byte: 0b01010101 = 0x55 = 85
      assertThat(encoded).hasSize(1);
      assertThat(encoded[0] & 0xFF).isEqualTo(0x55);
    }

    @Test
    void signBitEncoding_zeroBecomesOne() {
      // 0 >= 0 is true, so zero maps to 1
      float[] vector = {0.0f, -1.0f, 0.0f, 1.0f};
      var dataset = new ArrayVectorDataset(new float[][] {vector});
      var bq = BinaryQuantizer.train(dataset, BinaryMode.SIGN_BIT);

      byte[] encoded = bq.encode(vector);
      // Bits: d0=1(0>=0), d1=0, d2=1(0>=0), d3=1 → 0b1101 = 0x0D = 13
      assertThat(encoded[0] & 0xFF).isEqualTo(0x0D);
    }

    @Test
    void encodeAndDecode_roundTrip_producesSignVector() {
      float[] vector = {1.5f, -2.3f, 0.7f, -0.1f, 3.0f, -1.0f, 0.0f, 2.2f};
      var dataset = new ArrayVectorDataset(new float[][] {vector});
      var bq = BinaryQuantizer.train(dataset, BinaryMode.SIGN_BIT);

      byte[] encoded = bq.encode(vector);
      float[] decoded = bq.decode(encoded);

      // SIGN_BIT decode: bit=1 → +1, bit=0 → -1
      float[] expected = {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f};
      assertThat(decoded).hasSize(8);
      for (int i = 0; i < 8; i++) {
        assertThat(decoded[i]).isCloseTo(expected[i], within(1e-6f));
      }
    }

    @Test
    void encodedByteSize_equalsCeilDimDiv8() {
      // dim=128: 128/8=16 bytes
      var bq = BinaryQuantizer.train(makeDataset(128), BinaryMode.SIGN_BIT);
      assertThat(bq.encodedByteSize()).isEqualTo(16);

      // dim=100: ceil(100/8)=13 bytes
      var bq2 = BinaryQuantizer.train(makeDataset(100), BinaryMode.SIGN_BIT);
      assertThat(bq2.encodedByteSize()).isEqualTo(13);

      // dim=1: ceil(1/8)=1 byte
      var bq3 = BinaryQuantizer.train(makeDataset(1), BinaryMode.SIGN_BIT);
      assertThat(bq3.encodedByteSize()).isEqualTo(1);
    }

    @Test
    void compressionRatio_is32x() {
      var bq = BinaryQuantizer.train(makeDataset(128), BinaryMode.SIGN_BIT);
      // 128 * 4 bytes / 16 bytes = 32.0
      assertThat(bq.compressionRatio()).isCloseTo(32.0f, within(0.01f));
    }

    @Test
    void encode_reusedDstBuffer_doesNotCorruptBits() {
      var bq = BinaryQuantizer.train(makeDataset(16), BinaryMode.SIGN_BIT);
      byte[] dst = new byte[bq.encodedByteSize()];

      // First encode: all positive → all 1-bits
      float[] allPos = new float[16];
      java.util.Arrays.fill(allPos, 1.0f);
      bq.encode(allPos, dst);
      // All bits should be 1 → 0xFF per byte
      for (byte b : dst) {
        assertThat(b & 0xFF).isEqualTo(0xFF);
      }

      // Second encode with same buffer: all negative → all 0-bits
      float[] allNeg = new float[16];
      java.util.Arrays.fill(allNeg, -1.0f);
      bq.encode(allNeg, dst);
      // All bits should be 0
      for (byte b : dst) {
        assertThat(b & 0xFF).isEqualTo(0x00);
      }
    }

    @Test
    void encode_dimensionMismatch_throws() {
      var bq = BinaryQuantizer.train(makeDataset(64), BinaryMode.SIGN_BIT);
      assertThatThrownBy(() -> bq.encode(new float[32]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Expected dimension 64");
    }
  }

  @Nested
  @Tag("unit")
  class BbqEncoding {

    @Test
    void bbqEncoding_centersBeforeSignBit() {
      // Centroid of [[2, -2], [4, -4]] = [3, -3]
      // Centered: [2-3, -2-(-3)] = [-1, 1] → bits: [0, 1]
      //           [4-3, -4-(-3)] = [1, -1] → bits: [1, 0]
      float[][] vectors = {{2.0f, -2.0f}, {4.0f, -4.0f}};
      var dataset = new ArrayVectorDataset(vectors);
      var bbq = BinaryQuantizer.train(dataset, BinaryMode.BBQ);

      byte[] enc0 = bbq.encode(vectors[0]);
      byte[] enc1 = bbq.encode(vectors[1]);

      // Vector [2, -2] centered to [-1, 1]: d0=0, d1=1 → 0b10 = 2
      assertThat(enc0[0] & 0xFF).isEqualTo(0x02);
      // Vector [4, -4] centered to [1, -1]: d0=1, d1=0 → 0b01 = 1
      assertThat(enc1[0] & 0xFF).isEqualTo(0x01);
    }

    @Test
    void bbqCorrections_distToCEqualsSquaredNorm() {
      float[][] vectors = generateVectors(100, 64, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var bbq = BinaryQuantizer.train(dataset, BinaryMode.BBQ);
      float[] centroid = bbq.centroid();

      var compressed = bbq.encodeAll(dataset);
      for (int i = 0; i < 10; i++) {
        float[] corrections = compressed.getCorrections(i);
        // Compute expected distToC manually
        float expectedDist = 0f;
        for (int d = 0; d < 64; d++) {
          float v = vectors[i][d] - centroid[d];
          expectedDist += v * v;
        }
        assertThat(corrections[0]).isCloseTo(expectedDist, within(1e-3f));
      }
    }

    @Test
    void bbqCorrections_vlIsMinCenteredComponent() {
      float[][] vectors = generateVectors(50, 32, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var bbq = BinaryQuantizer.train(dataset, BinaryMode.BBQ);
      float[] centroid = bbq.centroid();

      var compressed = bbq.encodeAll(dataset);
      for (int i = 0; i < 10; i++) {
        float[] corrections = compressed.getCorrections(i);
        float expectedMin = Float.POSITIVE_INFINITY;
        for (int d = 0; d < 32; d++) {
          float v = vectors[i][d] - centroid[d];
          if (v < expectedMin) expectedMin = v;
        }
        assertThat(corrections[1]).isCloseTo(expectedMin, within(1e-5f));
      }
    }

    @Test
    void bbqCorrections_widthIsMaxMinusMinCenteredComponent() {
      float[][] vectors = generateVectors(50, 32, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var bbq = BinaryQuantizer.train(dataset, BinaryMode.BBQ);
      float[] centroid = bbq.centroid();

      var compressed = bbq.encodeAll(dataset);
      for (int i = 0; i < 10; i++) {
        float[] corrections = compressed.getCorrections(i);
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (int d = 0; d < 32; d++) {
          float v = vectors[i][d] - centroid[d];
          if (v < min) min = v;
          if (v > max) max = v;
        }
        assertThat(corrections[2]).isCloseTo(max - min, within(1e-5f));
      }
    }

    @Test
    void bbqEncoding_centroidIsDatasetMean() {
      float[][] vectors = {{1.0f, 3.0f, 5.0f}, {3.0f, 5.0f, 7.0f}, {2.0f, 4.0f, 6.0f}};
      var dataset = new ArrayVectorDataset(vectors);
      var bbq = BinaryQuantizer.train(dataset, BinaryMode.BBQ);

      float[] centroid = bbq.centroid();
      // Mean: [(1+3+2)/3, (3+5+4)/3, (5+7+6)/3] = [2, 4, 6]
      assertThat(centroid[0]).isCloseTo(2.0f, within(1e-5f));
      assertThat(centroid[1]).isCloseTo(4.0f, within(1e-5f));
      assertThat(centroid[2]).isCloseTo(6.0f, within(1e-5f));
    }

    @Test
    void encode_reusedDstBuffer_doesNotCorruptBits() {
      float[][] vectors = generateVectors(100, 64, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var bbq = BinaryQuantizer.train(dataset, BinaryMode.BBQ);

      byte[] dst = new byte[bbq.encodedByteSize()];

      // Encode vector 0
      bbq.encode(vectors[0], dst);
      byte[] first = java.util.Arrays.copyOf(dst, dst.length);

      // Encode vector 1 into same buffer
      bbq.encode(vectors[1], dst);

      // Re-encode vector 0 and verify it matches first encoding
      bbq.encode(vectors[0], dst);
      assertThat(dst).isEqualTo(first);
    }

    @Test
    void bbqDecode_addsBackCentroid() {
      // Centroid of [[2, -2], [4, -4]] = [3, -3].
      // Centered vector 0: [-1, 1] → bits [0, 1] → decoded ±1 → [-1, +1] → add centroid → [2, -2].
      // Centered vector 1: [+1, -1] → bits [1, 0] → decoded ±1 → [+1, -1] → add centroid → [4, -4].
      float[][] vectors = {{2.0f, -2.0f}, {4.0f, -4.0f}};
      var dataset = new ArrayVectorDataset(vectors);
      var bbq = BinaryQuantizer.train(dataset, BinaryMode.BBQ);

      float[] decoded0 = bbq.decode(bbq.encode(vectors[0]));
      float[] decoded1 = bbq.decode(bbq.encode(vectors[1]));

      // Decoded vectors should match originals exactly for this symmetric 2-vector dataset.
      assertThat(decoded0[0]).isCloseTo(2.0f, within(1e-5f));
      assertThat(decoded0[1]).isCloseTo(-2.0f, within(1e-5f));
      assertThat(decoded1[0]).isCloseTo(4.0f, within(1e-5f));
      assertThat(decoded1[1]).isCloseTo(-4.0f, within(1e-5f));
    }

    @Test
    void compressionRatio_bbq_accountsForCorrectionBytes() {
      // dim=128: bit codes = 16 bytes, corrections = 12 bytes → total = 28 bytes
      // ratio = (128 * 4) / 28 = 512 / 28 ≈ 18.29
      var bbq = BinaryQuantizer.train(makeDataset(128), BinaryMode.BBQ);
      float expected = (128 * 4.0f) / (16 + 12);
      assertThat(bbq.compressionRatio()).isCloseTo(expected, within(0.01f));
      // Sanity: BBQ ratio must be strictly less than SIGN_BIT's 32x
      var signBit = BinaryQuantizer.train(makeDataset(128), BinaryMode.SIGN_BIT);
      assertThat(bbq.compressionRatio()).isLessThan(signBit.compressionRatio());
    }
  }

  @Nested
  @Tag("unit")
  class SignBitScoring {

    @Test
    void hammingDistance_identicalVectors_scoreIsOne() {
      float[] vector = {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};
      float[][] data = {vector};
      var dataset = new ArrayVectorDataset(data);
      var bq = BinaryQuantizer.train(dataset, BinaryMode.SIGN_BIT);
      var compressed = bq.encodeAll(dataset);

      ScoreFunction sf = compressed.scoreFunctionFor(vector, SimilarityFunction.DOT_PRODUCT);
      assertThat(sf.score(0)).isCloseTo(1.0f, within(1e-6f));
    }

    @Test
    void hammingDistance_oppositeVectors_scoreIsZero() {
      float[] pos = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
      float[] neg = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
      float[][] data = {neg};
      var dataset = new ArrayVectorDataset(data);
      var bq = BinaryQuantizer.train(dataset, BinaryMode.SIGN_BIT);
      var compressed = bq.encodeAll(dataset);

      ScoreFunction sf = compressed.scoreFunctionFor(pos, SimilarityFunction.DOT_PRODUCT);
      // All bits differ → hamming = dim → score = 1 - dim/dim = 0
      assertThat(sf.score(0)).isCloseTo(0.0f, within(1e-6f));
    }

    @Test
    void hammingDistance_orthogonalVectors_scoreIsHalf() {
      // Half of bits match, half differ
      float[] query = {1, 1, 1, 1, -1, -1, -1, -1};
      float[] stored = {1, 1, -1, -1, 1, 1, -1, -1};
      // Hamming = 4 (dims 2,3,4,5 differ) → score = 1 - 4/8 = 0.5
      float[][] data = {stored};
      var dataset = new ArrayVectorDataset(data);
      var bq = BinaryQuantizer.train(dataset, BinaryMode.SIGN_BIT);
      var compressed = bq.encodeAll(dataset);

      ScoreFunction sf = compressed.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
      assertThat(sf.score(0)).isCloseTo(0.5f, within(1e-6f));
    }

    @ParameterizedTest
    @EnumSource(SimilarityFunction.class)
    void rankCorrelatesWithTrueScore(SimilarityFunction simFunc) {
      float[][] vectors = generateNormalizedVectors(200, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var bq = BinaryQuantizer.train(dataset, BinaryMode.SIGN_BIT);
      var compressed = bq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction sf = compressed.scoreFunctionFor(query, simFunc);

      // Compute approximate and true scores
      float[] approxScores = new float[vectors.length];
      float[] trueScores = new float[vectors.length];
      for (int i = 0; i < vectors.length; i++) {
        approxScores[i] = sf.score(i);
        trueScores[i] = simFunc.compare(query, vectors[i]);
      }

      // Check top-10 overlap: SIGN_BIT on normalized 128-dim vectors should find ≥4 of true top 10
      int[] approxTop10 = topK(approxScores, 10);
      int[] trueTop10 = topK(trueScores, 10);
      int overlap = countOverlap(approxTop10, trueTop10);
      assertThat(overlap).as("SIGN_BIT %s: top-10 overlap", simFunc).isGreaterThanOrEqualTo(4);
    }
  }

  @Nested
  @Tag("unit")
  class BbqScoring {

    @Test
    void bbqScore_betterThanSimpleBq_onRandomData() {
      float[][] vectors = generateNormalizedVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);

      var signBit = BinaryQuantizer.train(dataset, BinaryMode.SIGN_BIT);
      var bbq = BinaryQuantizer.train(dataset, BinaryMode.BBQ);
      var signBitCompressed = signBit.encodeAll(dataset);
      var bbqCompressed = bbq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction sfSign = signBitCompressed.scoreFunctionFor(query, SimilarityFunction.COSINE);
      ScoreFunction sfBbq = bbqCompressed.scoreFunctionFor(query, SimilarityFunction.COSINE);

      // Compute true scores
      float[] trueScores = new float[vectors.length];
      float[] signScores = new float[vectors.length];
      float[] bbqScores = new float[vectors.length];
      for (int i = 0; i < vectors.length; i++) {
        trueScores[i] = SimilarityFunction.COSINE.compare(query, vectors[i]);
        signScores[i] = sfSign.score(i);
        bbqScores[i] = sfBbq.score(i);
      }

      // BBQ should have better rank correlation than SIGN_BIT
      int[] trueTop20 = topK(trueScores, 20);
      int[] signTop20 = topK(signScores, 20);
      int[] bbqTop20 = topK(bbqScores, 20);

      int signOverlap = countOverlap(trueTop20, signTop20);
      int bbqOverlap = countOverlap(trueTop20, bbqTop20);
      // BBQ must be at least as good as SIGN_BIT in rank correlation…
      assertThat(bbqOverlap)
          .as("BBQ should have better or equal top-20 overlap than SIGN_BIT")
          .isGreaterThanOrEqualTo(signOverlap);
      // …and must also meet an absolute floor so the test catches BBQ regressions
      // independently of how SIGN_BIT performs (e.g. both can't silently be 0).
      assertThat(bbqOverlap).as("BBQ top-20 overlap (absolute floor)").isGreaterThanOrEqualTo(8);
    }

    @ParameterizedTest
    @EnumSource(SimilarityFunction.class)
    void bbqRankCorrelation(SimilarityFunction simFunc) {
      float[][] vectors = generateNormalizedVectors(300, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var bbq = BinaryQuantizer.train(dataset, BinaryMode.BBQ);
      var compressed = bbq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction sf = compressed.scoreFunctionFor(query, simFunc);

      float[] approxScores = new float[vectors.length];
      float[] trueScores = new float[vectors.length];
      for (int i = 0; i < vectors.length; i++) {
        approxScores[i] = sf.score(i);
        trueScores[i] = simFunc.compare(query, vectors[i]);
      }

      // BBQ on normalized 128-dim vectors should find ≥5 of true top 10 (>50% recall@10)
      int[] approxTop10 = topK(approxScores, 10);
      int[] trueTop10 = topK(trueScores, 10);
      int overlap = countOverlap(approxTop10, trueTop10);
      assertThat(overlap).as("BBQ %s: top-10 overlap", simFunc).isGreaterThanOrEqualTo(5);
    }

    @Test
    void asymmetricDot_matchesNaiveComputation() {
      // Create known stored bits and query int4 values, compute manually
      int dim = 64;
      int numLongs = 1;

      // Stored: first 32 bits are 1, rest are 0
      long[] stored = {0x00000000FFFFFFFFL};

      // Query int4: dimension d has value (d % 16)
      byte[] q4 = new byte[dim];
      for (int d = 0; d < dim; d++) {
        q4[d] = (byte) (d % 16);
      }

      long[][] bitPlanes = BinaryQuantizedVectors.extractBitPlanes(q4, numLongs);
      int result = BinaryQuantizedVectors.asymmetricDot(stored, bitPlanes);

      // Manual: sum q4[d] for d in [0..31] where stored bit is 1
      int expected = 0;
      for (int d = 0; d < 32; d++) {
        expected += d % 16;
      }
      assertThat(result).isEqualTo(expected);
    }

    @Test
    void bitPlaneExtraction_correctness() {
      // Value 13 = 0b1101 → bit0=1, bit1=0, bit2=1, bit3=1
      byte[] q4 = {13, 0, 7, 15}; // 4-dim
      int numLongs = 1;

      long[][] planes = BinaryQuantizedVectors.extractBitPlanes(q4, numLongs);

      // Plane 0 (bit 0): dims with bit0=1: 13→1, 0→0, 7→1, 15→1 → 0b1101 = 13
      assertThat(planes[0][0]).isEqualTo(0b1101L);
      // Plane 1 (bit 1): dims with bit1=1: 13→0, 0→0, 7→1, 15→1 → 0b1100 = 12
      assertThat(planes[1][0]).isEqualTo(0b1100L);
      // Plane 2 (bit 2): dims with bit2=1: 13→1, 0→0, 7→1, 15→1 → 0b1101 = 13
      assertThat(planes[2][0]).isEqualTo(0b1101L);
      // Plane 3 (bit 3): dims with bit3=1: 13→1, 0→0, 7→0, 15→1 → 0b1001 = 9
      assertThat(planes[3][0]).isEqualTo(0b1001L);
    }
  }

  @Nested
  @Tag("unit")
  class BitPacking {

    @Test
    void bytesToLongs_roundTrip() {
      Random rng = new Random(42L);
      byte[] original = new byte[16]; // 128-bit = 2 longs
      rng.nextBytes(original);

      long[] longs = new long[2];
      BinaryQuantizer.bytesToLongs(original, longs);

      byte[] recovered = new byte[16];
      BinaryQuantizer.longsToBytes(longs, recovered);

      assertThat(recovered).isEqualTo(original);
    }

    @Test
    void packBits_dimensionNotMultipleOf64_handlesTail() {
      // dim=100: 13 bytes, 2 longs (128 bits, last 28 unused)
      float[] vector = new float[100];
      java.util.Arrays.fill(vector, 1.0f); // all positive → all bits set

      byte[] packed = new byte[13]; // ceil(100/8)
      BinaryQuantizer.packBits(vector, packed, null);

      long[] longs = new long[2];
      BinaryQuantizer.bytesToLongs(packed, longs);

      // First 64 bits should all be 1
      assertThat(longs[0]).isEqualTo(-1L); // 0xFFFFFFFFFFFFFFFF

      // Last 36 bits should be set (bits 0-35 of second long)
      long expectedSecondLong = (1L << 36) - 1;
      assertThat(longs[1]).isEqualTo(expectedSecondLong);
    }

    @Test
    void packBits_allPositive_allOnes() {
      float[] vector = new float[64];
      java.util.Arrays.fill(vector, 5.0f);

      byte[] packed = new byte[8]; // 64 bits = 8 bytes
      BinaryQuantizer.packBits(vector, packed, null);

      for (byte b : packed) {
        assertThat(b & 0xFF).isEqualTo(0xFF);
      }
    }

    @Test
    void packBits_allNegative_allZeros() {
      float[] vector = new float[64];
      java.util.Arrays.fill(vector, -5.0f);

      byte[] packed = new byte[8];
      BinaryQuantizer.packBits(vector, packed, null);

      for (byte b : packed) {
        assertThat(b & 0xFF).isEqualTo(0x00);
      }
    }
  }

  @Nested
  @Tag("slow")
  @EnabledIf("com.integrallis.vectors.quantization.BinaryQuantizerTest#siftAvailable")
  class SiftSmallBQ {

    @Test
    void signBit_recall_onSiftSmall() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_DIR.resolve("siftsmall_groundtruth.ivecs"));

      var dataset = SiftLoader.asDataset(base);
      var bq = BinaryQuantizer.train(dataset, BinaryMode.SIGN_BIT);
      var compressed = bq.encodeAll(dataset);

      double totalRecall = computeAverageRecall(compressed, base, queries, groundTruth, 10);
      assertThat(totalRecall)
          .as("SIGN_BIT average recall@10 on SIFT Small")
          .isGreaterThan(0.20); // SIFT is unnormalized, so BQ recall is lower
    }

    @Test
    void bbq_recall_onSiftSmall() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_DIR.resolve("siftsmall_groundtruth.ivecs"));

      var dataset = SiftLoader.asDataset(base);
      var bbq = BinaryQuantizer.train(dataset, BinaryMode.BBQ);
      var compressed = bbq.encodeAll(dataset);

      double totalRecall = computeAverageRecall(compressed, base, queries, groundTruth, 10);
      assertThat(totalRecall).as("BBQ average recall@10 on SIFT Small").isGreaterThan(0.30);
    }

    @Test
    void bbq_recall_betterThanSignBit_onSiftSmall() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_DIR.resolve("siftsmall_groundtruth.ivecs"));

      var dataset = SiftLoader.asDataset(base);

      var signBit = BinaryQuantizer.train(dataset, BinaryMode.SIGN_BIT);
      var signBitCompressed = signBit.encodeAll(dataset);
      double signBitRecall =
          computeAverageRecall(signBitCompressed, base, queries, groundTruth, 10);

      var bbq = BinaryQuantizer.train(dataset, BinaryMode.BBQ);
      var bbqCompressed = bbq.encodeAll(dataset);
      double bbqRecall = computeAverageRecall(bbqCompressed, base, queries, groundTruth, 10);

      assertThat(bbqRecall)
          .as("BBQ recall should be >= SIGN_BIT recall on SIFT Small")
          .isGreaterThanOrEqualTo(signBitRecall);
    }

    private double computeAverageRecall(
        BinaryQuantizedVectors compressed,
        float[][] base,
        float[][] queries,
        int[][] groundTruth,
        int k) {
      double totalRecall = 0;
      int numQueries = Math.min(10, queries.length);
      for (int q = 0; q < numQueries; q++) {
        ScoreFunction scoreFunc =
            compressed.scoreFunctionFor(queries[q], SimilarityFunction.EUCLIDEAN);

        float[] approxScores = new float[base.length];
        for (int i = 0; i < base.length; i++) {
          approxScores[i] = scoreFunc.score(i);
        }

        int[] approxTopK = topK(approxScores, k);
        int[] trueTopK = new int[k];
        System.arraycopy(groundTruth[q], 0, trueTopK, 0, k);

        totalRecall += SiftLoader.recallAtK(trueTopK, approxTopK, k);
      }
      return totalRecall / numQueries;
    }
  }

  // --- Helper methods ---

  private static ArrayVectorDataset makeDataset(int dim) {
    float[][] vectors = new float[1][dim];
    return new ArrayVectorDataset(vectors);
  }

  private static float[][] generateVectors(int count, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] vectors = new float[count][dim];
    for (int i = 0; i < count; i++) {
      for (int d = 0; d < dim; d++) {
        vectors[i][d] = rng.nextFloat() * 2 - 1; // [-1, 1]
      }
    }
    return vectors;
  }

  private static float[][] generateNormalizedVectors(int count, int dim, long seed) {
    float[][] vectors = generateVectors(count, dim, seed);
    for (float[] v : vectors) l2normalize(v);
    return vectors;
  }

  private static void l2normalize(float[] v) {
    float norm = 0;
    for (float f : v) norm += f * f;
    norm = (float) Math.sqrt(norm);
    if (norm > 0) {
      for (int d = 0; d < v.length; d++) v[d] /= norm;
    }
  }

  private static int[] topK(float[] scores, int k) {
    int[] indices = new int[k];
    boolean[] used = new boolean[scores.length];
    for (int i = 0; i < k; i++) {
      int best = -1;
      float bestScore = Float.NEGATIVE_INFINITY;
      for (int j = 0; j < scores.length; j++) {
        if (!used[j] && scores[j] > bestScore) {
          bestScore = scores[j];
          best = j;
        }
      }
      indices[i] = best;
      used[best] = true;
    }
    return indices;
  }

  private static int countOverlap(int[] a, int[] b) {
    int count = 0;
    for (int ai : a) {
      for (int bi : b) {
        if (ai == bi) {
          count++;
          break;
        }
      }
    }
    return count;
  }
}
