/**
 * Off-heap memory management, mmap file access, and arena-based storage.
 *
 * <p>Uses {@link java.lang.foreign.MemorySegment} and {@link java.lang.foreign.Arena} for
 * zero-copy, off-heap vector storage with spatial and temporal safety.
 */
package com.integrallis.vectors.storage;
