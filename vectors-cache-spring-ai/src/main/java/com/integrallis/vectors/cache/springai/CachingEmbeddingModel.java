package com.integrallis.vectors.cache.springai;

import com.integrallis.vectors.cache.VectorCache;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.UnaryOperator;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;

/**
 * Caching decorator around a Spring AI {@link EmbeddingModel}. All text-based entry points are
 * served from the supplied {@link VectorCache}; misses are forwarded to the delegate and the result
 * is stored under the normalized key.
 *
 * <p>Keys are derived from the raw input text through the configured {@link UnaryOperator}
 * normalizer (identity by default). Supply a normalizer that trims/lowercases/hashes to control
 * collision semantics for your workload. The batched {@link #embed(List)} path is vectorised: each
 * text is looked up individually, misses are accumulated into a single delegate call, and the
 * results are written back in order.
 *
 * <p>{@link #embed(List, org.springframework.ai.embedding.EmbeddingOptions,
 * org.springframework.ai.embedding.BatchingStrategy)} is not cached because it depends on per-call
 * options and batching policy; it forwards to the delegate.
 */
public class CachingEmbeddingModel implements EmbeddingModel {

  private static final Logger LOG = Logger.getLogger(CachingEmbeddingModel.class.getName());

  private final EmbeddingModel delegate;
  private final VectorCache<String, float[]> cache;
  private final UnaryOperator<String> keyFn;
  private final int dimensions;

  /**
   * @see #CachingEmbeddingModel(EmbeddingModel, VectorCache, UnaryOperator)
   */
  public CachingEmbeddingModel(EmbeddingModel delegate, VectorCache<String, float[]> cache) {
    this(delegate, cache, UnaryOperator.identity());
  }

  /**
   * @param delegate non-null underlying model
   * @param cache non-null key/value cache keyed by normalized text
   * @param keyFn text-normalization function applied before cache lookup and store
   */
  public CachingEmbeddingModel(
      EmbeddingModel delegate, VectorCache<String, float[]> cache, UnaryOperator<String> keyFn) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
    this.cache = Objects.requireNonNull(cache, "cache");
    this.keyFn = Objects.requireNonNull(keyFn, "keyFn");
    this.dimensions = detectDimensions(delegate);
  }

  /**
   * @return the wrapped delegate
   */
  public final EmbeddingModel delegate() {
    return delegate;
  }

  @Override
  public EmbeddingResponse call(EmbeddingRequest request) {
    Objects.requireNonNull(request, "request");
    List<String> texts = request.getInstructions();
    List<float[]> vectors = embedBatch(texts);
    List<Embedding> out = new ArrayList<>(vectors.size());
    for (int i = 0; i < vectors.size(); i++) {
      out.add(new Embedding(vectors.get(i), i));
    }
    return new EmbeddingResponse(out);
  }

  @Override
  public float[] embed(String text) {
    Objects.requireNonNull(text, "text");
    return cache.getOrCompute(keyFn.apply(text), k -> delegate.embed(text)).clone();
  }

  @Override
  public float[] embed(Document document) {
    Objects.requireNonNull(document, "document");
    String text = document.getText();
    String raw = text == null ? "" : text;
    return cache.getOrCompute(keyFn.apply(raw), k -> delegate.embed(document)).clone();
  }

  @Override
  public List<float[]> embed(List<String> texts) {
    Objects.requireNonNull(texts, "texts");
    return embedBatch(texts);
  }

  @Override
  public EmbeddingResponse embedForResponse(List<String> texts) {
    return call(new EmbeddingRequest(texts, null));
  }

  @Override
  public int dimensions() {
    return dimensions > 0 ? dimensions : delegate.dimensions();
  }

  /**
   * Batched embedding path that coalesces cache misses into a single delegate round-trip. Results
   * are emitted in request order; misses are stored under their normalized keys after the delegate
   * returns.
   */
  private List<float[]> embedBatch(List<String> texts) {
    int n = texts.size();
    float[][] out = new float[n][];
    List<Integer> missIndexes = new ArrayList<>();
    List<String> missTexts = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      String text = Objects.requireNonNull(texts.get(i), "texts[i]");
      String key = keyFn.apply(text);
      var hit = cache.get(key);
      if (hit.isPresent()) {
        out[i] = hit.get().clone();
      } else {
        missIndexes.add(i);
        missTexts.add(text);
      }
    }
    if (!missTexts.isEmpty()) {
      List<float[]> fresh = delegate.embed(missTexts);
      if (fresh.size() != missTexts.size()) {
        throw new IllegalStateException(
            "delegate returned "
                + fresh.size()
                + " embeddings for "
                + missTexts.size()
                + " inputs");
      }
      for (int j = 0; j < fresh.size(); j++) {
        int i = missIndexes.get(j);
        float[] v = fresh.get(j);
        out[i] = v;
        cache.put(keyFn.apply(missTexts.get(j)), v);
      }
    }
    List<float[]> list = new ArrayList<>(n);
    for (float[] v : out) {
      list.add(v);
    }
    return list;
  }

  private static int detectDimensions(EmbeddingModel model) {
    try {
      return model.dimensions();
    } catch (RuntimeException e) {
      LOG.log(
          Level.WARNING,
          "failed to detect dimensions from delegate; dimensions() will delegate on each call",
          e);
      return -1;
    }
  }
}
