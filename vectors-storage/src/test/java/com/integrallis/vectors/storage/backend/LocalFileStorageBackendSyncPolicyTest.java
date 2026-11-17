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
package com.integrallis.vectors.storage.backend;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The sync policy decides whether the backend forces every write to the disk. {@code EVERY_WRITE}
 * (the default) forces the object file and its directory on every put and delete. {@code
 * REPLICATED} forces nothing and leaves durability to whoever replicates the object elsewhere; the
 * atomic rename still rules out a torn object. Both read back the same bytes and etags.
 */
class LocalFileStorageBackendSyncPolicyTest {

  @TempDir Path tmp;

  @Test
  void everyWriteForcesAndReplicatedDoesNot() throws Exception {
    AtomicInteger forcedDefault = new AtomicInteger();
    LocalFileStorageBackend forcing =
        new LocalFileStorageBackend(
            tmp.resolve("forcing"),
            counting(forcedDefault),
            LocalFileStorageBackend.SyncPolicy.EVERY_WRITE);
    AtomicInteger forcedReplicated = new AtomicInteger();
    LocalFileStorageBackend replicated =
        new LocalFileStorageBackend(
            tmp.resolve("replicated"),
            counting(forcedReplicated),
            LocalFileStorageBackend.SyncPolicy.REPLICATED);
    byte[] v1 = "one".getBytes(StandardCharsets.UTF_8);
    byte[] v2 = "two".getBytes(StandardCharsets.UTF_8);
    for (LocalFileStorageBackend b : new LocalFileStorageBackend[] {forcing, replicated}) {
      b.put("coll/a", v1);
      StorageBackend.StoredValue a = b.getWithEtag("coll/a");
      assertThat(a.value()).isEqualTo(v1);
      StorageBackend.ConditionalPutResult cas = b.conditionalPut("coll/a", v2, a.etag());
      assertThat(cas.succeeded()).isTrue();
      assertThat(b.get("coll/a")).isEqualTo(v2);
      assertThat(b.conditionalPut("coll/a", v1, a.etag()).succeeded()).isFalse();
      assertThat(b.list("coll/")).containsExactly("coll/a");
      b.delete("coll/a");
      assertThat(b.get("coll/a")).isNull();
    }
    assertThat(forcing.getWithEtag("x")).isNull();
    assertThat(forcedDefault.get()).as("the default forces on every write").isPositive();
    assertThat(forcedReplicated.get()).as("replicated durability forces nothing").isZero();
    // the one-argument constructor is the forcing one
    assertThat(new LocalFileStorageBackend(tmp.resolve("plain")).syncPolicy())
        .isEqualTo(LocalFileStorageBackend.SyncPolicy.EVERY_WRITE);
  }

  private static LocalFileStorageBackend.ChannelOpener counting(AtomicInteger forces) {
    return (path, options) -> new CountingChannel(FileChannel.open(path, options), forces);
  }

  /** Real file IO; counts the forces. */
  private static final class CountingChannel extends FileChannel {
    private final FileChannel delegate;
    private final AtomicInteger forces;

    CountingChannel(FileChannel delegate, AtomicInteger forces) {
      this.delegate = delegate;
      this.forces = forces;
    }

    @Override
    public void force(boolean metadata) throws IOException {
      forces.incrementAndGet();
      delegate.force(metadata);
    }

    @Override
    protected void implCloseChannel() throws IOException {
      delegate.close();
    }

    @Override
    public int read(ByteBuffer dst) throws IOException {
      return delegate.read(dst);
    }

    @Override
    public long read(ByteBuffer[] dsts, int offset, int length) throws IOException {
      return delegate.read(dsts, offset, length);
    }

    @Override
    public int write(ByteBuffer src) throws IOException {
      return delegate.write(src);
    }

    @Override
    public long write(ByteBuffer[] srcs, int offset, int length) throws IOException {
      return delegate.write(srcs, offset, length);
    }

    @Override
    public long position() throws IOException {
      return delegate.position();
    }

    @Override
    public FileChannel position(long newPosition) throws IOException {
      delegate.position(newPosition);
      return this;
    }

    @Override
    public long size() throws IOException {
      return delegate.size();
    }

    @Override
    public FileChannel truncate(long size) throws IOException {
      delegate.truncate(size);
      return this;
    }

    @Override
    public long transferTo(long position, long count, WritableByteChannel target)
        throws IOException {
      return delegate.transferTo(position, count, target);
    }

    @Override
    public long transferFrom(ReadableByteChannel src, long position, long count)
        throws IOException {
      return delegate.transferFrom(src, position, count);
    }

    @Override
    public int read(ByteBuffer dst, long position) throws IOException {
      return delegate.read(dst, position);
    }

    @Override
    public int write(ByteBuffer src, long position) throws IOException {
      return delegate.write(src, position);
    }

    @Override
    public MappedByteBuffer map(MapMode mode, long position, long size) throws IOException {
      return delegate.map(mode, position, size);
    }

    @Override
    public FileLock lock(long position, long size, boolean shared) throws IOException {
      return delegate.lock(position, size, shared);
    }

    @Override
    public FileLock tryLock(long position, long size, boolean shared) throws IOException {
      return delegate.tryLock(position, size, shared);
    }
  }
}
