/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.vectors.ingest;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Internal collector that buffers {@link EmbeddedDoc}s into successive {@link Batch}es according to
 * a {@link BatchPolicy}. Not thread-safe — the {@link IngestPipeline} owns a single accumulator and
 * drives it from a single commit thread.
 *
 * <p>Three orthogonal triggers flush a batch: doc count, accumulated text+blob bytes, or wall time
 * since the first doc was added (checked via {@link #flushIfTimedOut(long)}).
 */
final class BatchAccumulator {

  private final BatchPolicy policy;
  private final long maxLatencyNanos;
  private final List<EmbeddedDoc> buffer;
  private long bytesAccumulated;
  private long firstAddedAtNanos;
  private long nextBatchId;

  BatchAccumulator(BatchPolicy policy) {
    this.policy = Objects.requireNonNull(policy, "policy");
    this.maxLatencyNanos = policy.maxLatency().toNanos();
    this.buffer = new ArrayList<>(Math.min(policy.maxDocs(), 1024));
  }

  /**
   * Adds a doc. If this push trips a count or size threshold, returns the resulting flushed batch;
   * otherwise empty.
   */
  Optional<Batch> add(EmbeddedDoc doc) {
    Objects.requireNonNull(doc, "doc");
    if (buffer.isEmpty()) {
      firstAddedAtNanos = System.nanoTime();
    }
    buffer.add(doc);
    bytesAccumulated += sizeOf(doc.doc());
    if (buffer.size() >= policy.maxDocs() || bytesAccumulated >= policy.maxBytes()) {
      return Optional.of(flushInternal());
    }
    return Optional.empty();
  }

  /**
   * Returns a batch if the buffer is non-empty and {@code nowNanos} is at least {@code maxLatency}
   * past the first add; otherwise empty.
   */
  Optional<Batch> flushIfTimedOut(long nowNanos) {
    if (buffer.isEmpty()) return Optional.empty();
    if (nowNanos - firstAddedAtNanos < maxLatencyNanos) return Optional.empty();
    return Optional.of(flushInternal());
  }

  /** Forces a flush of any remaining docs (used at shutdown). Empty when buffer is empty. */
  Optional<Batch> drain() {
    if (buffer.isEmpty()) return Optional.empty();
    return Optional.of(flushInternal());
  }

  /** Number of docs currently buffered (never includes already-flushed batches). */
  int pendingDocs() {
    return buffer.size();
  }

  long pendingBytes() {
    return bytesAccumulated;
  }

  private Batch flushInternal() {
    Batch out = new Batch(nextBatchId++, List.copyOf(buffer));
    buffer.clear();
    bytesAccumulated = 0L;
    firstAddedAtNanos = 0L;
    return out;
  }

  /** Bytes attributable to a doc — text (UTF-8) + blob; vectors are excluded per BatchPolicy. */
  static long sizeOf(IngestDoc d) {
    long s = 0;
    String text = d.text();
    if (text != null) s += utf8ByteLength(text);
    byte[] blob = d.blob();
    if (blob != null) s += blob.length;
    return s;
  }

  /**
   * Computes the number of bytes {@code text} would occupy when encoded as UTF-8, in a single pass
   * without allocating an intermediate {@code byte[]} (as {@code getBytes(UTF_8)} would). The
   * result is bit-identical to {@code text.getBytes(StandardCharsets.UTF_8).length}: unpaired
   * surrogates are counted as one byte each, matching the encoder's single-byte {@code '?'}
   * replacement.
   */
  static long utf8ByteLength(String text) {
    long count = 0;
    int len = text.length();
    for (int i = 0; i < len; i++) {
      char c = text.charAt(i);
      if (c < 0x80) {
        count += 1;
      } else if (c < 0x800) {
        count += 2;
      } else if (Character.isHighSurrogate(c)) {
        if (i + 1 < len && Character.isLowSurrogate(text.charAt(i + 1))) {
          count += 4; // valid surrogate pair -> one 4-byte code point
          i++;
        } else {
          count += 1; // unpaired high surrogate -> '?' replacement
        }
      } else if (Character.isLowSurrogate(c)) {
        count += 1; // unpaired low surrogate -> '?' replacement
      } else {
        count += 3;
      }
    }
    return count;
  }
}
