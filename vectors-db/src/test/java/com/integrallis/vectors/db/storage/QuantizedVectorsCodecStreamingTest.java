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

import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.quantization.ArrayVectorDataset;
import com.integrallis.vectors.quantization.CompressedVectors;
import com.integrallis.vectors.quantization.ExtendedRaBitQuantizedVectors;
import com.integrallis.vectors.quantization.ExtendedRaBitQuantizer;
import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Proves the streaming {@code quantized.bin} codec (Ext-RaBitQ) matches the {@code byte[]} codec
 * and round-trips through a chunked stream — the >2 GB (~50 GB at 100M) codes path.
 */
@Tag("unit")
class QuantizedVectorsCodecStreamingTest {

  private record Fixture(ExtendedRaBitQuantizedVectors codes, ExtendedRaBitQuantizer quant) {}

  private static Fixture sample() {
    int n = 1500;
    int dim = 64;
    Random r = new Random(3L);
    float[][] vecs = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        vecs[i][d] = r.nextFloat() * 2f - 1f;
      }
    }
    ArrayVectorDataset ds = new ArrayVectorDataset(vecs);
    ExtendedRaBitQuantizer quant = ExtendedRaBitQuantizer.train(ds, 4);
    return new Fixture(quant.encodeAll(ds), quant);
  }

  @Test
  void streamingEncodeIsByteIdenticalToArrayEncode() throws Exception {
    Fixture f = sample();
    byte[] arrayForm =
        QuantizedVectorsCodec.encode(f.codes, f.quant, QuantizerKind.EXTENDED_RABITQ);

    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    long written =
        QuantizedVectorsCodec.encode(f.codes, f.quant, QuantizerKind.EXTENDED_RABITQ, bos);

    assertThat(written).isEqualTo(arrayForm.length);
    assertThat(bos.toByteArray()).isEqualTo(arrayForm);
  }

  @Test
  void streamingDecodeMatchesArrayDecode() throws Exception {
    Fixture f = sample();
    byte[] form = QuantizedVectorsCodec.encode(f.codes, f.quant, QuantizerKind.EXTENDED_RABITQ);

    CompressedVectors viaArray = QuantizedVectorsCodec.decode(form);
    CompressedVectors viaStream = QuantizedVectorsCodec.decode(new ByteArrayInputStream(form));
    assertCodesEqual(
        (ExtendedRaBitQuantizedVectors) viaArray, (ExtendedRaBitQuantizedVectors) viaStream);
    assertCodesEqual(f.codes, (ExtendedRaBitQuantizedVectors) viaStream);
  }

  @Test
  void chunkedRoundTripThroughStorage() throws Exception {
    Fixture f = sample();
    HeapStorageBackend backend = new HeapStorageBackend();
    String base = "gen-0000000000000001/" + FileFormat.QUANTIZED_FILE;

    // 700-byte chunks force many objects.
    long total =
        ChunkedBlob.writeStream(
            backend,
            base,
            os -> QuantizedVectorsCodec.encode(f.codes, f.quant, QuantizerKind.EXTENDED_RABITQ, os),
            700);
    assertThat(total)
        .isEqualTo(
            QuantizedVectorsCodec.encode(f.codes, f.quant, QuantizerKind.EXTENDED_RABITQ).length);
    assertThat(backend.list("gen-0000000000000001/").stream().filter(k -> k.contains(base + ".")))
        .as("codes shipped as multiple chunk objects")
        .hasSizeGreaterThan(1);

    try (InputStream in = ChunkedBlob.openStream(backend, base)) {
      CompressedVectors decoded = QuantizedVectorsCodec.decode(in);
      assertCodesEqual(f.codes, (ExtendedRaBitQuantizedVectors) decoded);
    }
  }

  private static void assertCodesEqual(
      ExtendedRaBitQuantizedVectors a, ExtendedRaBitQuantizedVectors b) {
    assertThat(b.size()).isEqualTo(a.size());
    assertThat(b.dimension()).isEqualTo(a.dimension());
    for (int i = 0; i < a.size(); i++) {
      assertThat(b.getSignCodes(i)).as("signCodes " + i).isEqualTo(a.getSignCodes(i));
      assertThat(b.getMagCodes(i)).as("magCodes " + i).isEqualTo(a.getMagCodes(i));
      assertThat(b.getCorrections(i)).as("corrections " + i).isEqualTo(a.getCorrections(i));
    }
  }
}
