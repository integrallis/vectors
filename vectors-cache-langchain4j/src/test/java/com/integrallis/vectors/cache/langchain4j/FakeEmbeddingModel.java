package com.integrallis.vectors.cache.langchain4j;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/** Deterministic test embedding model: trigram hash, 8 dims. */
final class FakeEmbeddingModel implements EmbeddingModel {

  static final int DIM = 8;

  final AtomicInteger singleCalls = new AtomicInteger();
  final AtomicInteger batchCalls = new AtomicInteger();

  @Override
  public Response<Embedding> embed(String text) {
    singleCalls.incrementAndGet();
    return Response.from(Embedding.from(encode(text)));
  }

  @Override
  public Response<Embedding> embed(TextSegment segment) {
    return embed(segment.text());
  }

  @Override
  public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
    batchCalls.incrementAndGet();
    List<Embedding> out = new ArrayList<>(segments.size());
    for (TextSegment s : segments) {
      out.add(Embedding.from(encode(s.text())));
    }
    return Response.from(out);
  }

  @Override
  public int dimension() {
    return DIM;
  }

  static float[] encode(String text) {
    float[] v = new float[DIM];
    for (int i = 0; i + 2 < text.length(); i++) {
      int h = (text.charAt(i) * 31 + text.charAt(i + 1)) * 31 + text.charAt(i + 2);
      v[Math.floorMod(h, DIM)] += 1.0f;
    }
    return v;
  }
}
