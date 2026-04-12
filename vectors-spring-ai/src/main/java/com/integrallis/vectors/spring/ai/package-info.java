/**
 * Spring AI {@link org.springframework.ai.vectorstore.VectorStore} adapter backed by java-vectors.
 *
 * <p>Provides {@link com.integrallis.vectors.spring.ai.JavaVectorsVectorStore}, a drop-in
 * replacement for Spring AI's {@code SimpleVectorStore} that uses HNSW/Vamana graph indexing,
 * SIMD-accelerated distance kernels, optional quantization, and mmap persistence.
 *
 * <p>Requires JDK 25+.
 */
package com.integrallis.vectors.spring.ai;
