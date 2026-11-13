# Audit: Does Models Embedding Backend Reach Grouped Batched Matmul APIs? (#186)

**Hypothesis:** The Models `GgufEmbeddingBackend` never directly calls grouped (dual/triple)
batched matmul APIs. Encoder-only models (BERT, MiniLM) extract hidden states directly without
a generative forward pass. Decoder models repurposed for embedding (GTE-Qwen2, E5-Mistral) run
the full forward pass internally, which *does* use grouped projection — but this happens inside
Models, not at the Vectors API boundary.

**Decision rule:** If grouped batched matmul calls are observed during a Model embedding run,
the hypothesis is rejected and we document which model architectures trigger them. If no grouped
calls are observed, we file a Models issue to understand why grouped projection is not reached or
whether the embedding path bypasses the transformer layers.

## Source-code audit

### Models `GgufEmbeddingBackend.embed(String)`

```java
float[] embed(String text) {
    backend.reset();
    if (modelOwnsPooling) {
        return backend.embedSequence(tokens);     // encoder: hidden state extraction
    }
    switch (pooling) {
        case MEAN -> meanPooled(tokens);          // per-position hidden state, average
        case CLS -> clsPooled(tokens);            // position-0 hidden state
        case LAST_TOKEN -> lastTokenPooled(tokens); // final position hidden state
    }
}
```

None of these paths call `TensorOps.ggufBatchedMatmul` or any grouped variant directly. They
call `PureJavaBackend.embedSequence()` / `hiddenState()` / `prefillHiddenState()`, which delegate
to the model-specific forward pass.

### Models `GgufEmbeddingBackend.embedAll(List<String>)`

```java
float[][] embedAll(List<String> texts) {
    if (pooling == LAST_TOKEN && backend.supportsBatchedHiddenStates() && texts.size() > 1) {
        return batchedEmbedLastToken(texts);     // calls backend.prefillBatchHiddenStates()
    }
    // MEAN and CLS: fall back to sequential per-document embed()
    for (String text : texts) rows[i] = embed(text);
}
```

Again, no direct Vectors matmul calls. The batched prefill path runs inside the forward pass
(`LlamaForwardPass.runSessionPrefillRows()`, `Qwen35ForwardPass.prefillBatchHiddenStates()`,
etc.).

### Forward pass grouped projection

The grouped (dual/triple) batched matmul APIs are called only in generative forward passes:

| Forward Pass | Grouped call site | Matrices grouped |
|---|---|---|
| `LlamaForwardPass.multiplyDualProjection()` | `TensorOps.ggufDualBatchedMatmul()` | gate + up (FFN) |
| `LlamaForwardPass.multiplyTripleProjection()` | `TensorOps.ggufTripleBatchedMatmul()` | query + key + value (QKV) |
| `Qwen35ForwardPass.multiplyDualProjection()` | `TensorOps.ggufDualBatchedMatmul()` | gate + up (FFN) |
| `Qwen35ForwardPass.multiplyTripleProjection()` | `TensorOps.ggufTripleBatchedMatmul()` | query + key + value (QKV) |

`TensorOps.ggufDualBatchedMatmul` / `ggufTripleBatchedMatmul` then call the Vectors grouped
APIs (`VectorUtil.ggufQ4_KQ8_KDualBatchedMatmul`, `ggufQ8_0Q8_0TripleBatchedMatmul`, etc.) when
formats match.

### Conclusion from source audit

1. **Encoder-only embedding models** (BERT, MiniLM, sentence-transformers) → NO grouped
   projection. They extract hidden states directly from encoder layers; no generative forward
   pass runs.

2. **Decoder-based embedding models** (GTE-Qwen2, E5-Mistral) → YES, grouped projection IS
   reached. These models run the full generative forward pass internally, including grouped QKV
   and FFN projections. But this happens inside Models, not at the Vectors API boundary that
   the embedding backend controls.

3. **The embedding backend never calls grouped APIs directly.** Whether grouped projection is
   reached depends entirely on which `PureJavaDecoder` implementation handles the model, not on
   the embedding backend code.

## Runtime verification

After implementing `GgufBatchedMatmulKernel.recordGrouped()` in Vectors, run an embedding
benchmark in Models and check `GgufBatchedMatmulKernel.groupedCounts(format)`:

```bash
# In a Models integration test that loads a decoder-based embedding model (e.g., GTE-Qwen2)
# and calls embedAll() with multiple documents:
./gradlew :backend-java:test --tests "com.integrallis.models.backend.purejava.EmbeddingGemma2BackendRouteTest"
```

If the model is decoder-based and uses Q4_K/Q8_0 weights, `groupedCounts()` should show:
- `Q4_K`: groupedCalls > 0 (from grouped QKV / FFN dual projection inside the forward pass)
- `Q8_0`: groupedCalls > 0 (same reason)

If the model is encoder-only (e.g., BERT), `groupedCounts()` should all be zero.

## Implications for #118 (item 5)

The issue asks: "Determine whether Models exposes independent-text batching and reaches the
appropriate existing primitive."

**Answer:**
- For LAST_TOKEN pooling on decoder models: YES, `embedAll` reaches `prefillBatchHiddenStates()`
  which internally uses batched (and grouped) matmul through the forward pass.
- For MEAN/CLS pooling: NO, Models falls back to sequential `embed()` per document. The
  `prefillBatchHiddenStates()` fast path is LAST_TOKEN only.

This is a **Models design decision**, not a Vectors missing primitive. Vectors already provides
the primitives (`ggufDualBatchedMatmul`, `ggufTripleBatchedMatmul`); whether Models reaches
them depends on which forward pass handles the model. The grouped call counters added for #186
let us verify this at runtime rather than inferring from source code.

## Deliverable status

- [x] Source-code audit of Models → Vectors call paths
- [x] `recordGrouped()` added to `GgufBatchedMatmulKernel`
- [ ] Runtime verification with actual decoder-based embedding model
- [ ] Runtime verification with actual encoder-only embedding model
