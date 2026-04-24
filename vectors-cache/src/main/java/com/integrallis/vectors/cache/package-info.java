/**
 * Pluggable caching layer for AI workloads that use {@code java-vectors}.
 *
 * <p>Two minimal SPIs — {@link com.integrallis.vectors.cache.VectorCache VectorCache} and {@link
 * com.integrallis.vectors.cache.SemanticCache SemanticCache} — cover exact-match and similarity-
 * search lookups. A Caffeine-backed default is bundled; JCache, semantic-on-VectorCollection, and
 * framework-specific decorators are opt-in satellite modules.
 */
package com.integrallis.vectors.cache;
