package com.integrallis.vectors.demo.langchain4j;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.util.List;
import java.util.Objects;

/**
 * Zero-dependency, deterministic LangChain4j {@link EmbeddingModel}.
 *
 * <p>Bag-of-character-trigrams vector of the configured dimension, L2-normalized. Similar texts
 * produce similar vectors — enough to make retrieval feel meaningful without requiring model
 * downloads.
 *
 * <p>Not for production. In a real application, substitute e.g. {@code AllMiniLmL6V2EmbeddingModel}
 * ({@code langchain4j-embeddings-all-minilm-l6-v2}) or {@code OpenAiEmbeddingModel}.
 */
final class DeterministicEmbeddingModel implements EmbeddingModel {

  private final int dimension;

  DeterministicEmbeddingModel(int dimension) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("dimension must be positive: " + dimension);
    }
    this.dimension = dimension;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public Response<Embedding> embed(String text) {
    Objects.requireNonNull(text, "text");
    return Response.from(Embedding.from(trigramVector(text)));
  }

  @Override
  public Response<Embedding> embed(TextSegment textSegment) {
    return embed(textSegment.text());
  }

  @Override
  public Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {
    List<Embedding> out =
        textSegments.stream().map(s -> Embedding.from(trigramVector(s.text()))).toList();
    return Response.from(out);
  }

  private float[] trigramVector(String text) {
    float[] out = new float[dimension];
    String normalized = text.toLowerCase();
    if (normalized.length() < 3) {
      normalized = "  " + normalized + "  ";
    }
    for (int i = 0; i <= normalized.length() - 3; i++) {
      int h =
          31 * (31 * normalized.charAt(i) + normalized.charAt(i + 1)) + normalized.charAt(i + 2);
      int idx = Math.floorMod(h, dimension);
      out[idx] += 1.0f;
    }
    float norm = 0f;
    for (float v : out) {
      norm += v * v;
    }
    norm = (float) Math.sqrt(norm);
    if (norm > 0f) {
      for (int i = 0; i < dimension; i++) {
        out[i] /= norm;
      }
    }
    return out;
  }
}
