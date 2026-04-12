/**
 * LangChain4J {@link dev.langchain4j.store.embedding.EmbeddingStore} adapter backed by
 * java-vectors.
 *
 * <p>Provides {@link com.integrallis.vectors.langchain4j.JavaVectorsEmbeddingStore}, a drop-in
 * replacement for LangChain4J's {@code InMemoryEmbeddingStore} that uses HNSW/Vamana graph
 * indexing, SIMD-accelerated distance kernels, optional quantization, and mmap persistence.
 *
 * <p>Requires JDK 25+.
 */
package com.integrallis.vectors.langchain4j;
