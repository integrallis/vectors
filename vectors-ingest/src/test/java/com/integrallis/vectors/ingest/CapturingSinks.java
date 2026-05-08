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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/** Tiny in-memory sinks/embedder used by pipeline tests. Package-private. */
final class CapturingSinks {

  private CapturingSinks() {}

  static final class CapturingVectorSink implements VectorSink {
    final List<Batch> staged = Collections.synchronizedList(new ArrayList<>());
    final List<Batch> committed = Collections.synchronizedList(new ArrayList<>());
    private final AtomicLong committedCount = new AtomicLong();
    volatile long addAllSleepMs;
    volatile boolean failOnNextAddAll;

    @Override
    public void addAll(Batch batch) throws IOException {
      if (failOnNextAddAll) {
        failOnNextAddAll = false;
        throw new IOException("simulated addAll failure");
      }
      if (addAllSleepMs > 0) {
        try {
          Thread.sleep(addAllSleepMs);
        } catch (InterruptedException ie) {
          Thread.currentThread().interrupt();
          throw new IOException("interrupted", ie);
        }
      }
      staged.add(batch);
    }

    @Override
    public void commit() {
      // last staged → committed
      synchronized (staged) {
        if (!staged.isEmpty()) {
          Batch b = staged.remove(staged.size() - 1);
          committed.add(b);
          committedCount.addAndGet(b.size());
        }
      }
    }

    @Override
    public long committedCount() {
      return committedCount.get();
    }
  }

  static final class CapturingSidecartSink implements SidecartSink {
    final List<Batch> received = Collections.synchronizedList(new ArrayList<>());
    private final AtomicLong written = new AtomicLong();

    @Override
    public void writeAll(Batch batch) {
      received.add(batch);
      written.addAndGet(batch.size());
    }

    @Override
    public long writtenCount() {
      return written.get();
    }
  }

  /** Embedder that returns a vector encoding the doc id length plus a per-doc index. */
  static final class FakeEmbedder implements Embedder {
    private final int dim;
    final List<Integer> calls = Collections.synchronizedList(new ArrayList<>());
    volatile boolean failNext;

    FakeEmbedder(int dim) {
      this.dim = dim;
    }

    @Override
    public String name() {
      return "fake";
    }

    @Override
    public int dimension() {
      return dim;
    }

    @Override
    public List<float[]> embedAll(List<IngestDoc> docs) {
      if (failNext) {
        failNext = false;
        throw new IllegalArgumentException("simulated embed failure");
      }
      calls.add(docs.size());
      List<float[]> out = new ArrayList<>(docs.size());
      for (IngestDoc d : docs) {
        float[] v = new float[dim];
        v[0] = d.id().length();
        out.add(v);
      }
      return out;
    }
  }
}
