package com.integrallis.vectors.db.storage;

import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.quantization.BinaryMode;
import com.integrallis.vectors.quantization.BinaryQuantizedVectors;
import com.integrallis.vectors.quantization.BinaryQuantizer;
import com.integrallis.vectors.quantization.CompressedVectors;
import com.integrallis.vectors.quantization.GivensRotation;
import com.integrallis.vectors.quantization.NVQuantizedVectors;
import com.integrallis.vectors.quantization.NVQuantizer;
import com.integrallis.vectors.quantization.PQVectors;
import com.integrallis.vectors.quantization.ProductQuantizer;
import com.integrallis.vectors.quantization.Quantizer;
import com.integrallis.vectors.quantization.QuaternionRotation;
import com.integrallis.vectors.quantization.RaBitQuantizedVectors;
import com.integrallis.vectors.quantization.RaBitQuantizer;
import com.integrallis.vectors.quantization.RandomRotation;
import com.integrallis.vectors.quantization.Rotation;
import com.integrallis.vectors.quantization.ScalarBits;
import com.integrallis.vectors.quantization.ScalarQuantizedVectors;
import com.integrallis.vectors.quantization.ScalarQuantizer;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Binary codec for the {@code quantized.bin} file. Serializes and deserializes all 6 quantizer
 * kinds (SQ8, SQ4, PQ, BQ, RABITQ, NVQ) using a tagged-union wire format.
 *
 * <p>Layout (little-endian throughout):
 *
 * <pre>
 * Offset  Size     Field                       Notes
 * ------  ------   --------------------------  --------------------------------
 *   0      4       magic                       FileFormat.MAGIC_QUANTIZED
 *   4      4       format version              FileFormat.VERSION_QUANTIZED (= 1)
 *   8      4       quantizerKind ordinal       QuantizerKind.ordinal()
 *  12      4       dimension                   original vector dimension
 *  16      4       vectorCount                 number of compressed vectors
 *  20      varies  quantizer state             trained quantizer parameters
 *         varies  compressed vector data       per-vector encoded bytes
 * </pre>
 *
 * <p><b>CRC coverage.</b> Like {@code graph.bin}, this file is not self-checksummed — its CRC lives
 * in the {@linkplain Manifest#quantizedBinCrc32() manifest}.
 */
public final class QuantizedVectorsCodec {

  /** Common header size: magic(4) + version(4) + kind(4) + dimension(4) + vectorCount(4). */
  static final int COMMON_HEADER_SIZE = 20;

  /** Rotation type tags for serialization. */
  private static final int ROTATION_TAG_RANDOM = 0;

  private static final int ROTATION_TAG_GIVENS = 1;
  private static final int ROTATION_TAG_QUATERNION = 2;

  private QuantizedVectorsCodec() {}

  /**
   * Serializes compressed vectors and their quantizer into a byte array matching the quantized.bin
   * format.
   *
   * @param compressed the compressed vector data
   * @param quantizer the trained quantizer that produced the compressed vectors
   * @param kind the quantizer kind (must not be NONE)
   * @return the serialized bytes
   * @throws IllegalArgumentException if kind is NONE or the quantizer/compressed types don't match
   */
  public static byte[] encode(
      CompressedVectors compressed, Quantizer<?> quantizer, QuantizerKind kind) {
    Objects.requireNonNull(compressed, "compressed must not be null");
    Objects.requireNonNull(quantizer, "quantizer must not be null");
    Objects.requireNonNull(kind, "kind must not be null");
    if (kind == QuantizerKind.NONE) {
      throw new IllegalArgumentException("Cannot encode NONE quantizer kind");
    }

    return switch (kind) {
      case SQ8, SQ4 ->
          encodeSQ((ScalarQuantizedVectors) compressed, (ScalarQuantizer) quantizer, kind);
      case PQ -> encodePQ((PQVectors) compressed, (ProductQuantizer) quantizer);
      case BQ -> encodeBQ((BinaryQuantizedVectors) compressed, (BinaryQuantizer) quantizer);
      case RABITQ -> encodeRaBitQ((RaBitQuantizedVectors) compressed, (RaBitQuantizer) quantizer);
      case NVQ -> encodeNVQ((NVQuantizedVectors) compressed, (NVQuantizer) quantizer);
      case NONE -> throw new AssertionError("unreachable");
    };
  }

  /**
   * Deserializes a {@code quantized.bin} byte array into compressed vectors. The returned {@link
   * CompressedVectors} contains its own quantizer reference, ready for scoring.
   *
   * @param bytes the raw quantized.bin bytes
   * @return the deserialized compressed vectors
   * @throws IOException if the bytes are truncated, wrong magic/version, or encode inconsistent
   *     data
   */
  public static CompressedVectors decode(byte[] bytes) throws IOException {
    Objects.requireNonNull(bytes, "bytes must not be null");
    if (bytes.length < COMMON_HEADER_SIZE) {
      throw new IOException(
          "quantized.bin truncated: expected at least "
              + COMMON_HEADER_SIZE
              + " header bytes, got "
              + bytes.length);
    }

    ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);

    int magic = buf.getInt();
    if (magic != FileFormat.MAGIC_QUANTIZED) {
      throw new IOException(
          String.format(
              "quantized.bin magic mismatch: expected 0x%08x, got 0x%08x",
              FileFormat.MAGIC_QUANTIZED, magic));
    }
    int version = buf.getInt();
    if (version != FileFormat.VERSION_QUANTIZED) {
      throw new IOException(
          "quantized.bin version mismatch: expected "
              + FileFormat.VERSION_QUANTIZED
              + ", got "
              + version);
    }

    int kindOrdinal = buf.getInt();
    QuantizerKind[] kinds = QuantizerKind.values();
    if (kindOrdinal < 0 || kindOrdinal >= kinds.length) {
      throw new IOException("quantized.bin invalid quantizerKind ordinal: " + kindOrdinal);
    }
    QuantizerKind kind = kinds[kindOrdinal];
    if (kind == QuantizerKind.NONE) {
      throw new IOException("quantized.bin cannot have quantizerKind NONE");
    }

    int dimension = buf.getInt();
    if (dimension <= 0) {
      throw new IOException("quantized.bin dimension must be positive, got " + dimension);
    }

    int vectorCount = buf.getInt();
    if (vectorCount < 0) {
      throw new IOException("quantized.bin vectorCount must be >= 0, got " + vectorCount);
    }

    return switch (kind) {
      case SQ8, SQ4 -> decodeSQ(buf, kind, dimension, vectorCount);
      case PQ -> decodePQ(buf, dimension, vectorCount);
      case BQ -> decodeBQ(buf, dimension, vectorCount);
      case RABITQ -> decodeRaBitQ(buf, dimension, vectorCount);
      case NVQ -> decodeNVQ(buf, dimension, vectorCount);
      case NONE -> throw new AssertionError("unreachable");
    };
  }

  // ---- SQ8/SQ4 ----

  private static byte[] encodeSQ(
      ScalarQuantizedVectors compressed, ScalarQuantizer quantizer, QuantizerKind kind) {
    int dimension = quantizer.dimension();
    int vectorCount = compressed.size();
    ScalarBits bits = quantizer.bits();
    int encodedByteSize = bits.encodedByteSize(dimension);

    // Quantizer state: bits ordinal(4) + minQuantile(4) + maxQuantile(4) = 12 bytes
    int quantizerStateSize = 12;
    // Per-vector: encodedByteSize bytes + 1 float correction
    long vectorDataSize = (long) vectorCount * (encodedByteSize + Float.BYTES);
    long totalSize = COMMON_HEADER_SIZE + quantizerStateSize + vectorDataSize;
    checkSize(totalSize);

    byte[] out = new byte[(int) totalSize];
    ByteBuffer buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);

    writeCommonHeader(buf, kind, dimension, vectorCount);

    // Quantizer state
    buf.putInt(bits.ordinal());
    buf.putFloat(quantizer.minQuantile());
    buf.putFloat(quantizer.maxQuantile());

    // Per-vector data
    for (int i = 0; i < vectorCount; i++) {
      buf.put(compressed.getQuantizedVector(i));
      buf.putFloat(compressed.getCorrection(i));
    }

    return out;
  }

  private static ScalarQuantizedVectors decodeSQ(
      ByteBuffer buf, QuantizerKind kind, int dimension, int vectorCount) throws IOException {
    ensureRemaining(buf, 12, "SQ quantizer state");

    int bitsOrdinal = buf.getInt();
    ScalarBits[] allBits = ScalarBits.values();
    if (bitsOrdinal < 0 || bitsOrdinal >= allBits.length) {
      throw new IOException("quantized.bin invalid ScalarBits ordinal: " + bitsOrdinal);
    }
    ScalarBits bits = allBits[bitsOrdinal];

    // Validate kind matches bits
    if (kind == QuantizerKind.SQ8 && bits != ScalarBits.INT8) {
      throw new IOException("quantized.bin SQ8 kind but ScalarBits is " + bits);
    }
    if (kind == QuantizerKind.SQ4 && bits != ScalarBits.INT4) {
      throw new IOException("quantized.bin SQ4 kind but ScalarBits is " + bits);
    }

    float minQuantile = buf.getFloat();
    float maxQuantile = buf.getFloat();

    ScalarQuantizer quantizer =
        ScalarQuantizer.fromQuantiles(dimension, bits, minQuantile, maxQuantile);

    int encodedByteSize = bits.encodedByteSize(dimension);
    long perVector = encodedByteSize + Float.BYTES;
    ensureRemaining(buf, perVector * vectorCount, "SQ vector data");

    byte[][] quantizedVectors = new byte[vectorCount][encodedByteSize];
    float[] corrections = new float[vectorCount];
    for (int i = 0; i < vectorCount; i++) {
      buf.get(quantizedVectors[i]);
      corrections[i] = buf.getFloat();
    }

    return new ScalarQuantizedVectors(quantizer, quantizedVectors, corrections, dimension);
  }

  // ---- PQ ----

  private static byte[] encodePQ(PQVectors compressed, ProductQuantizer quantizer) {
    int dimension = quantizer.dimension();
    int vectorCount = compressed.size();
    int numSubspaces = quantizer.numSubspaces();
    int numClusters = quantizer.numClusters();
    float[] globalCentroid = quantizer.globalCentroid();
    boolean hasGlobalCentroid = globalCentroid != null;

    // Quantizer state size:
    // numSubspaces(4) + numClusters(4) + hasGlobalCentroid(4)
    // + globalCentroid(D*4 if present)
    // + subspaceSizesAndOffsets(M*2*4)
    // + codebooks (sum over m: numClusters * subDim_m * 4)
    int[][] sizesAndOffsets = quantizer.subspaceSizesAndOffsets();
    long quantizerStateSize = 12; // 3 ints
    if (hasGlobalCentroid) {
      quantizerStateSize += (long) dimension * Float.BYTES;
    }
    quantizerStateSize += (long) numSubspaces * 2 * Integer.BYTES;
    for (int m = 0; m < numSubspaces; m++) {
      int subDim = sizesAndOffsets[m][0];
      quantizerStateSize += (long) numClusters * subDim * Float.BYTES;
    }

    // Per-vector: numSubspaces bytes (one cluster index per subspace)
    long vectorDataSize = (long) vectorCount * numSubspaces;
    long totalSize = COMMON_HEADER_SIZE + quantizerStateSize + vectorDataSize;
    checkSize(totalSize);

    byte[] out = new byte[(int) totalSize];
    ByteBuffer buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);

    writeCommonHeader(buf, QuantizerKind.PQ, dimension, vectorCount);

    // Quantizer state
    buf.putInt(numSubspaces);
    buf.putInt(numClusters);
    buf.putInt(hasGlobalCentroid ? 1 : 0);

    if (hasGlobalCentroid) {
      for (int d = 0; d < dimension; d++) {
        buf.putFloat(globalCentroid[d]);
      }
    }

    for (int m = 0; m < numSubspaces; m++) {
      buf.putInt(sizesAndOffsets[m][0]); // size
      buf.putInt(sizesAndOffsets[m][1]); // offset
    }

    float[][] codebooks = quantizer.codebooks();
    for (int m = 0; m < numSubspaces; m++) {
      float[] codebook = codebooks[m];
      for (float v : codebook) {
        buf.putFloat(v);
      }
    }

    // Per-vector data
    for (int i = 0; i < vectorCount; i++) {
      buf.put(compressed.getCode(i));
    }

    return out;
  }

  private static PQVectors decodePQ(ByteBuffer buf, int dimension, int vectorCount)
      throws IOException {
    ensureRemaining(buf, 12, "PQ quantizer state header");

    int numSubspaces = buf.getInt();
    int numClusters = buf.getInt();
    int hasGlobalCentroidFlag = buf.getInt();

    if (numSubspaces <= 0) {
      throw new IOException("quantized.bin PQ numSubspaces must be positive: " + numSubspaces);
    }
    if (numClusters <= 0 || numClusters > 256) {
      throw new IOException("quantized.bin PQ numClusters must be in [1, 256]: " + numClusters);
    }

    float[] globalCentroid = null;
    if (hasGlobalCentroidFlag == 1) {
      ensureRemaining(buf, (long) dimension * Float.BYTES, "PQ globalCentroid");
      globalCentroid = new float[dimension];
      for (int d = 0; d < dimension; d++) {
        globalCentroid[d] = buf.getFloat();
      }
    } else if (hasGlobalCentroidFlag != 0) {
      throw new IOException(
          "quantized.bin PQ hasGlobalCentroid must be 0 or 1: " + hasGlobalCentroidFlag);
    }

    ensureRemaining(buf, (long) numSubspaces * 2 * Integer.BYTES, "PQ subspaceSizesAndOffsets");
    int[][] sizesAndOffsets = new int[numSubspaces][2];
    for (int m = 0; m < numSubspaces; m++) {
      sizesAndOffsets[m][0] = buf.getInt();
      sizesAndOffsets[m][1] = buf.getInt();
    }

    float[][] codebooks = new float[numSubspaces][];
    for (int m = 0; m < numSubspaces; m++) {
      int subDim = sizesAndOffsets[m][0];
      int codebookLen = numClusters * subDim;
      ensureRemaining(buf, (long) codebookLen * Float.BYTES, "PQ codebook " + m);
      codebooks[m] = new float[codebookLen];
      for (int j = 0; j < codebookLen; j++) {
        codebooks[m][j] = buf.getFloat();
      }
    }

    ProductQuantizer quantizer =
        ProductQuantizer.fromState(
            dimension, numSubspaces, numClusters, sizesAndOffsets, codebooks, globalCentroid);

    // Per-vector data
    ensureRemaining(buf, (long) vectorCount * numSubspaces, "PQ vector data");
    byte[][] codes = new byte[vectorCount][numSubspaces];
    for (int i = 0; i < vectorCount; i++) {
      buf.get(codes[i]);
    }

    return new PQVectors(quantizer, codes, dimension);
  }

  // ---- BQ ----

  private static byte[] encodeBQ(BinaryQuantizedVectors compressed, BinaryQuantizer quantizer) {
    int dimension = quantizer.dimension();
    int vectorCount = compressed.size();
    BinaryMode mode = quantizer.mode();
    float[] centroid = quantizer.centroid();
    int numLongs = (dimension + 63) >> 6;

    // Quantizer state: mode ordinal(4) + centroid (D*4 if BBQ, 0 if SIGN_BIT)
    long quantizerStateSize = 4;
    if (mode == BinaryMode.BBQ && centroid != null) {
      quantizerStateSize += (long) dimension * Float.BYTES;
    }

    // Per-vector: numLongs * 8 bytes + corrections (3 floats if BBQ, 0 if SIGN_BIT)
    int correctionsPerVector = (mode == BinaryMode.BBQ) ? 3 : 0;
    long perVector = (long) numLongs * Long.BYTES + (long) correctionsPerVector * Float.BYTES;
    long vectorDataSize = (long) vectorCount * perVector;
    long totalSize = COMMON_HEADER_SIZE + quantizerStateSize + vectorDataSize;
    checkSize(totalSize);

    byte[] out = new byte[(int) totalSize];
    ByteBuffer buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);

    writeCommonHeader(buf, QuantizerKind.BQ, dimension, vectorCount);

    // Quantizer state
    buf.putInt(mode.ordinal());
    if (mode == BinaryMode.BBQ && centroid != null) {
      for (int d = 0; d < dimension; d++) {
        buf.putFloat(centroid[d]);
      }
    }

    // Per-vector data
    for (int i = 0; i < vectorCount; i++) {
      long[] codes = compressed.getCode(i);
      for (long c : codes) {
        buf.putLong(c);
      }
      if (correctionsPerVector > 0) {
        float[] corrections = compressed.getCorrections(i);
        for (int j = 0; j < correctionsPerVector; j++) {
          buf.putFloat(corrections[j]);
        }
      }
    }

    return out;
  }

  private static BinaryQuantizedVectors decodeBQ(ByteBuffer buf, int dimension, int vectorCount)
      throws IOException {
    ensureRemaining(buf, 4, "BQ mode ordinal");

    int modeOrdinal = buf.getInt();
    BinaryMode[] modes = BinaryMode.values();
    if (modeOrdinal < 0 || modeOrdinal >= modes.length) {
      throw new IOException("quantized.bin invalid BinaryMode ordinal: " + modeOrdinal);
    }
    BinaryMode mode = modes[modeOrdinal];

    float[] centroid = null;
    if (mode == BinaryMode.BBQ) {
      ensureRemaining(buf, (long) dimension * Float.BYTES, "BQ centroid");
      centroid = new float[dimension];
      for (int d = 0; d < dimension; d++) {
        centroid[d] = buf.getFloat();
      }
    }

    BinaryQuantizer quantizer = BinaryQuantizer.fromState(dimension, mode, centroid);

    int numLongs = (dimension + 63) >> 6;
    int correctionsPerVector = (mode == BinaryMode.BBQ) ? 3 : 0;
    long perVector = (long) numLongs * Long.BYTES + (long) correctionsPerVector * Float.BYTES;
    ensureRemaining(buf, perVector * vectorCount, "BQ vector data");

    long[][] codes = new long[vectorCount][numLongs];
    float[][] corrections = (mode == BinaryMode.BBQ) ? new float[vectorCount][3] : null;
    for (int i = 0; i < vectorCount; i++) {
      for (int j = 0; j < numLongs; j++) {
        codes[i][j] = buf.getLong();
      }
      if (corrections != null) {
        for (int j = 0; j < 3; j++) {
          corrections[i][j] = buf.getFloat();
        }
      }
    }

    return new BinaryQuantizedVectors(quantizer, codes, corrections, dimension);
  }

  // ---- RaBitQ ----

  private static byte[] encodeRaBitQ(RaBitQuantizedVectors compressed, RaBitQuantizer quantizer) {
    int dimension = quantizer.dimension();
    int vectorCount = compressed.size();
    int paddedDimension = quantizer.paddedDimension();
    float[] centroid = quantizer.centroid();
    Rotation rotation = quantizer.rotation();
    int numLongs = paddedDimension >> 6;

    // Quantizer state: paddedDimension(4) + centroid(D*4) + rotation state
    long rotationSize = rotationEncodedSize(rotation);
    long quantizerStateSize = 4 + (long) dimension * Float.BYTES + rotationSize;

    // Per-vector: numLongs * 8 + 5 floats corrections
    long perVector = (long) numLongs * Long.BYTES + 5L * Float.BYTES;
    long vectorDataSize = (long) vectorCount * perVector;
    long totalSize = COMMON_HEADER_SIZE + quantizerStateSize + vectorDataSize;
    checkSize(totalSize);

    byte[] out = new byte[(int) totalSize];
    ByteBuffer buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);

    writeCommonHeader(buf, QuantizerKind.RABITQ, dimension, vectorCount);

    // Quantizer state
    buf.putInt(paddedDimension);
    for (int d = 0; d < dimension; d++) {
      buf.putFloat(centroid[d]);
    }
    encodeRotation(buf, rotation);

    // Per-vector data
    for (int i = 0; i < vectorCount; i++) {
      long[] codes = compressed.getCode(i);
      for (long c : codes) {
        buf.putLong(c);
      }
      float[] corrections = compressed.getCorrections(i);
      for (int j = 0; j < 5; j++) {
        buf.putFloat(corrections[j]);
      }
    }

    return out;
  }

  private static RaBitQuantizedVectors decodeRaBitQ(ByteBuffer buf, int dimension, int vectorCount)
      throws IOException {
    ensureRemaining(buf, 4, "RaBitQ paddedDimension");

    int paddedDimension = buf.getInt();
    if (paddedDimension <= 0 || paddedDimension % 64 != 0) {
      throw new IOException(
          "quantized.bin RaBitQ paddedDimension must be positive multiple of 64: "
              + paddedDimension);
    }

    ensureRemaining(buf, (long) dimension * Float.BYTES, "RaBitQ centroid");
    float[] centroid = new float[dimension];
    for (int d = 0; d < dimension; d++) {
      centroid[d] = buf.getFloat();
    }

    Rotation rotation = decodeRotation(buf, paddedDimension);

    RaBitQuantizer quantizer =
        RaBitQuantizer.fromState(dimension, paddedDimension, centroid, rotation);

    int numLongs = paddedDimension >> 6;
    long perVector = (long) numLongs * Long.BYTES + 5L * Float.BYTES;
    ensureRemaining(buf, perVector * vectorCount, "RaBitQ vector data");

    long[][] codes = new long[vectorCount][numLongs];
    float[][] corrections = new float[vectorCount][5];
    for (int i = 0; i < vectorCount; i++) {
      for (int j = 0; j < numLongs; j++) {
        codes[i][j] = buf.getLong();
      }
      for (int j = 0; j < 5; j++) {
        corrections[i][j] = buf.getFloat();
      }
    }

    return new RaBitQuantizedVectors(quantizer, codes, corrections, dimension);
  }

  // ---- NVQ ----

  private static byte[] encodeNVQ(NVQuantizedVectors compressed, NVQuantizer quantizer) {
    int dimension = quantizer.dimension();
    int vectorCount = compressed.size();
    int numSubvectors = quantizer.numSubvectors();
    int[] subvectorSizes = quantizer.subvectorSizes();
    float[] globalMean = quantizer.globalMean();

    // Quantizer state: numSubvectors(4) + subvectorSizes(M*4) + globalMean(D*4)
    long quantizerStateSize =
        4 + (long) numSubvectors * Integer.BYTES + (long) dimension * Float.BYTES;

    // Per-vector: dimension bytes (quantized) + numSubvectors * 4 floats (metadata)
    long perVector = dimension + (long) numSubvectors * 4 * Float.BYTES;
    long vectorDataSize = (long) vectorCount * perVector;
    long totalSize = COMMON_HEADER_SIZE + quantizerStateSize + vectorDataSize;
    checkSize(totalSize);

    byte[] out = new byte[(int) totalSize];
    ByteBuffer buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);

    writeCommonHeader(buf, QuantizerKind.NVQ, dimension, vectorCount);

    // Quantizer state
    buf.putInt(numSubvectors);
    for (int m = 0; m < numSubvectors; m++) {
      buf.putInt(subvectorSizes[m]);
    }
    for (int d = 0; d < dimension; d++) {
      buf.putFloat(globalMean[d]);
    }

    // Per-vector data
    for (int i = 0; i < vectorCount; i++) {
      buf.put(compressed.getQuantizedBytes(i));
      float[] metadata = compressed.getSubvectorMetadata(i);
      for (float v : metadata) {
        buf.putFloat(v);
      }
    }

    return out;
  }

  private static NVQuantizedVectors decodeNVQ(ByteBuffer buf, int dimension, int vectorCount)
      throws IOException {
    ensureRemaining(buf, 4, "NVQ numSubvectors");

    int numSubvectors = buf.getInt();
    if (numSubvectors <= 0) {
      throw new IOException("quantized.bin NVQ numSubvectors must be positive: " + numSubvectors);
    }

    ensureRemaining(buf, (long) numSubvectors * Integer.BYTES, "NVQ subvectorSizes");
    int[] subvectorSizes = new int[numSubvectors];
    for (int m = 0; m < numSubvectors; m++) {
      subvectorSizes[m] = buf.getInt();
    }

    ensureRemaining(buf, (long) dimension * Float.BYTES, "NVQ globalMean");
    float[] globalMean = new float[dimension];
    for (int d = 0; d < dimension; d++) {
      globalMean[d] = buf.getFloat();
    }

    NVQuantizer quantizer =
        NVQuantizer.fromState(dimension, numSubvectors, subvectorSizes, globalMean);

    int metadataFloats = numSubvectors * 4;
    long perVector = dimension + (long) metadataFloats * Float.BYTES;
    ensureRemaining(buf, perVector * vectorCount, "NVQ vector data");

    byte[][] quantizedBytes = new byte[vectorCount][dimension];
    float[][] subvectorMetadata = new float[vectorCount][metadataFloats];
    for (int i = 0; i < vectorCount; i++) {
      buf.get(quantizedBytes[i]);
      for (int j = 0; j < metadataFloats; j++) {
        subvectorMetadata[i][j] = buf.getFloat();
      }
    }

    return new NVQuantizedVectors(quantizer, quantizedBytes, subvectorMetadata, dimension);
  }

  // ---- Rotation serialization ----

  private static long rotationEncodedSize(Rotation rotation) {
    int dim = rotation.dimension();
    // type tag (4) + dimension (4) + type-specific data
    long base = 8;
    if (rotation instanceof RandomRotation) {
      // Q matrix (dim*dim*4) + Q^T matrix (dim*dim*4)
      return base + 2L * dim * dim * Float.BYTES;
    } else if (rotation instanceof GivensRotation) {
      // cos array (dim/2 * 4) + sin array (dim/2 * 4)
      int numPairs = dim / 2;
      return base + 2L * numPairs * Float.BYTES;
    } else if (rotation instanceof QuaternionRotation) {
      // qL (dim/4 * 4 * 4) + qR (dim/4 * 4 * 4)
      int numBlocks = dim / 4;
      return base + 2L * numBlocks * 4 * Float.BYTES;
    }
    throw new IllegalArgumentException("Unknown rotation type: " + rotation.getClass().getName());
  }

  private static void encodeRotation(ByteBuffer buf, Rotation rotation) {
    int dim = rotation.dimension();
    if (rotation instanceof RandomRotation rr) {
      buf.putInt(ROTATION_TAG_RANDOM);
      buf.putInt(dim);
      float[][] q = rr.matrix();
      float[][] qt = rr.matrixT();
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          buf.putFloat(q[i][j]);
        }
      }
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          buf.putFloat(qt[i][j]);
        }
      }
    } else if (rotation instanceof GivensRotation gr) {
      buf.putInt(ROTATION_TAG_GIVENS);
      buf.putInt(dim);
      float[] cos = gr.cos();
      float[] sin = gr.sin();
      for (float c : cos) {
        buf.putFloat(c);
      }
      for (float s : sin) {
        buf.putFloat(s);
      }
    } else if (rotation instanceof QuaternionRotation qr) {
      buf.putInt(ROTATION_TAG_QUATERNION);
      buf.putInt(dim);
      float[][] qL = qr.qL();
      float[][] qR = qr.qR();
      for (float[] q : qL) {
        for (float v : q) {
          buf.putFloat(v);
        }
      }
      for (float[] q : qR) {
        for (float v : q) {
          buf.putFloat(v);
        }
      }
    } else {
      throw new IllegalArgumentException("Unknown rotation type: " + rotation.getClass().getName());
    }
  }

  private static Rotation decodeRotation(ByteBuffer buf, int expectedDimension) throws IOException {
    ensureRemaining(buf, 8, "rotation header");
    int tag = buf.getInt();
    int dim = buf.getInt();
    if (dim != expectedDimension) {
      throw new IOException(
          "quantized.bin rotation dimension "
              + dim
              + " does not match expected "
              + expectedDimension);
    }

    return switch (tag) {
      case ROTATION_TAG_RANDOM -> {
        long matrixBytes = 2L * dim * dim * Float.BYTES;
        ensureRemaining(buf, matrixBytes, "RandomRotation matrices");
        float[][] q = new float[dim][dim];
        float[][] qt = new float[dim][dim];
        for (int i = 0; i < dim; i++) {
          for (int j = 0; j < dim; j++) {
            q[i][j] = buf.getFloat();
          }
        }
        for (int i = 0; i < dim; i++) {
          for (int j = 0; j < dim; j++) {
            qt[i][j] = buf.getFloat();
          }
        }
        yield RandomRotation.fromMatrix(dim, q, qt);
      }
      case ROTATION_TAG_GIVENS -> {
        int numPairs = dim / 2;
        ensureRemaining(buf, 2L * numPairs * Float.BYTES, "GivensRotation arrays");
        float[] cos = new float[numPairs];
        float[] sin = new float[numPairs];
        for (int i = 0; i < numPairs; i++) {
          cos[i] = buf.getFloat();
        }
        for (int i = 0; i < numPairs; i++) {
          sin[i] = buf.getFloat();
        }
        yield GivensRotation.fromCosSin(dim, cos, sin);
      }
      case ROTATION_TAG_QUATERNION -> {
        int numBlocks = dim / 4;
        ensureRemaining(buf, 2L * numBlocks * 4 * Float.BYTES, "QuaternionRotation arrays");
        float[][] qL = new float[numBlocks][4];
        float[][] qR = new float[numBlocks][4];
        for (int i = 0; i < numBlocks; i++) {
          for (int j = 0; j < 4; j++) {
            qL[i][j] = buf.getFloat();
          }
        }
        for (int i = 0; i < numBlocks; i++) {
          for (int j = 0; j < 4; j++) {
            qR[i][j] = buf.getFloat();
          }
        }
        yield QuaternionRotation.fromQuaternions(dim, qL, qR);
      }
      default -> throw new IOException("quantized.bin unknown rotation type tag: " + tag);
    };
  }

  // ---- Helpers ----

  private static void writeCommonHeader(
      ByteBuffer buf, QuantizerKind kind, int dimension, int vectorCount) {
    buf.putInt(FileFormat.MAGIC_QUANTIZED);
    buf.putInt(FileFormat.VERSION_QUANTIZED);
    buf.putInt(kind.ordinal());
    buf.putInt(dimension);
    buf.putInt(vectorCount);
  }

  private static void ensureRemaining(ByteBuffer buf, long needed, String what) throws IOException {
    if (buf.remaining() < needed) {
      throw new IOException(
          "quantized.bin truncated while reading "
              + what
              + ": need "
              + needed
              + " bytes, have "
              + buf.remaining());
    }
  }

  private static void checkSize(long totalSize) {
    if (totalSize > Integer.MAX_VALUE - 8L) {
      throw new IllegalStateException(
          "Encoded quantized.bin exceeds 2 GB limit: " + totalSize + " bytes");
    }
  }
}
