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

import static org.assertj.core.api.Assertions.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.*;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class LocalFileSyncFailureTest {
  @TempDir Path root;

  @Test
  void actualDirectoryForceFailurePropagatesAndClosesChannel() throws Exception {
    IOException failure = new java.nio.file.AccessDeniedException(root.toString());
    FileChannel channel =
        FileChannel.open(
            root.resolve("channel"), StandardOpenOption.CREATE, StandardOpenOption.WRITE);
    var backend =
        new LocalFileStorageBackend(
            root,
            (path, options) ->
                path.equals(root)
                    ? new ForceFailingChannel(channel, failure)
                    : FileChannel.open(path, options));
    assertThatThrownBy(() -> backend.put("entry", new byte[] {1})).isSameAs(failure);
    assertThat(channel.isOpen()).isFalse();
  }

  @Test
  void uncheckedDirectoryForceFailureAlsoPropagates() throws Exception {
    var failure =
        new java.io.UncheckedIOException(new IOException("injected unchecked sync failure"));
    FileChannel channel =
        FileChannel.open(
            root.resolve("channel"), StandardOpenOption.CREATE, StandardOpenOption.WRITE);
    var backend =
        new LocalFileStorageBackend(
            root,
            (path, options) ->
                path.equals(root)
                    ? new ForceFailingChannel(channel, failure)
                    : FileChannel.open(path, options));
    assertThatThrownBy(() -> backend.put("entry", new byte[] {1})).isSameAs(failure);
    assertThat(channel.isOpen()).isFalse();
  }

  @Test
  void valueFileForceFailurePropagatesBeforePublishing() throws Exception {
    new LocalFileStorageBackend(root).put("entry", new byte[] {7});
    IOException failure = new IOException("injected value force failure");
    var backend =
        new LocalFileStorageBackend(
            root,
            (path, options) -> new ForceFailingChannel(FileChannel.open(path, options), failure));
    assertThatThrownBy(() -> backend.put("entry", new byte[] {1})).isSameAs(failure);
    assertThat(backend.get("entry")).isEqualTo(new byte[] {7});
  }

  /** Real file IO with a controlled force failure; no instrumentation of JDK classes. */
  private static final class ForceFailingChannel extends FileChannel {
    private final FileChannel delegate;
    private final Exception failure;

    ForceFailingChannel(FileChannel delegate, Exception failure) {
      this.delegate = delegate;
      this.failure = failure;
    }

    @Override
    public void force(boolean metadata) throws IOException {
      if (failure instanceof IOException io) throw io;
      throw (RuntimeException) failure;
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
    public long read(ByteBuffer[] dst, int offset, int length) throws IOException {
      return delegate.read(dst, offset, length);
    }

    @Override
    public int write(ByteBuffer src) throws IOException {
      return delegate.write(src);
    }

    @Override
    public long write(ByteBuffer[] src, int offset, int length) throws IOException {
      return delegate.write(src, offset, length);
    }

    @Override
    public long position() throws IOException {
      return delegate.position();
    }

    @Override
    public FileChannel position(long p) throws IOException {
      delegate.position(p);
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
    public long transferTo(long p, long count, WritableByteChannel target) throws IOException {
      return delegate.transferTo(p, count, target);
    }

    @Override
    public long transferFrom(ReadableByteChannel src, long p, long count) throws IOException {
      return delegate.transferFrom(src, p, count);
    }

    @Override
    public int read(ByteBuffer dst, long p) throws IOException {
      return delegate.read(dst, p);
    }

    @Override
    public int write(ByteBuffer src, long p) throws IOException {
      return delegate.write(src, p);
    }

    @Override
    public MappedByteBuffer map(MapMode mode, long p, long size) throws IOException {
      return delegate.map(mode, p, size);
    }

    @Override
    public FileLock lock(long p, long size, boolean shared) throws IOException {
      return delegate.lock(p, size, shared);
    }

    @Override
    public FileLock tryLock(long p, long size, boolean shared) throws IOException {
      return delegate.tryLock(p, size, shared);
    }
  }
}
