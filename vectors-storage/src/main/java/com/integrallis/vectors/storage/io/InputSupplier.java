package com.integrallis.vectors.storage.io;

import java.io.IOException;

/**
 * Supplier of per-thread {@link RandomAccessInput} instances. Thread-safe; manages shared resources
 * (mapped segments, arenas).
 *
 * <p>Each call to {@link #open()} returns an independent reader with its own position state. The
 * supplier itself manages the shared underlying resource and should be closed when all readers are
 * done.
 */
public interface InputSupplier extends AutoCloseable {

  /** Creates a new independent reader. The caller is responsible for closing it. */
  RandomAccessInput open() throws IOException;

  /** Returns the total length of the underlying data source. */
  long length() throws IOException;

  @Override
  void close() throws IOException;
}
