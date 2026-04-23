package com.integrallis.vectors.vcr.testng;

/** Minimal stand-in for an LLM embedding model used by the listener tests. */
public interface FakeEmbeddingModel {
  float[] embed(String prompt);
}
