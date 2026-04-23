/**
 * Framework-neutral VCR (record/replay) engine for LLM embedding and chat calls.
 *
 * <p>Provides a storage-backed replacement for the Redis/Jedis/Gson VCR harness: cassettes are
 * persisted through {@link com.integrallis.vectors.storage.backend.StorageBackend} via {@link
 * com.integrallis.vectors.vcr.ExactCassetteStore}, and JSON serialization is pluggable via the
 * {@link com.integrallis.vectors.vcr.CassetteSerializer} SPI. Semantic (vector-similarity) lookup
 * is provided by the optional {@code vectors-vcr-semantic-db} module.
 */
package com.integrallis.vectors.vcr;
