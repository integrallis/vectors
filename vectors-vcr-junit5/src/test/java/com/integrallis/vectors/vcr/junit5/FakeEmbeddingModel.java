package com.integrallis.vectors.vcr.junit5;

/**
 * Tiny stand-in for an LLM embedding model used by the extension test. Exposes an {@code
 * embed(String)} method and returns a deterministic vector so recording and playback can be
 * verified without pulling in a real framework (LangChain4j / Spring AI).
 */
interface FakeEmbeddingModel {
  float[] embed(String prompt);
}
