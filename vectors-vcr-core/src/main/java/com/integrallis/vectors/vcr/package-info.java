/**
 * Framework-neutral VCR (record/replay) engine for LLM embedding and chat calls.
 *
 * <p>Provides a storage-backed replacement for the Redis/Jedis/Gson VCR harness: cassettes are
 * persisted through {@link com.integrallis.vectors.storage.backend.StorageBackend} (exact mode) or
 * {@link com.integrallis.vectors.db.VectorCollection} (semantic mode), and JSON serialization is
 * pluggable via the {@link com.integrallis.vectors.vcr.CassetteSerializer} SPI.
 */
package com.integrallis.vectors.vcr;
