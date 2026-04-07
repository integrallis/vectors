package com.integrallis.vectors.storage.io;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * {@link InputSupplier} that memory-maps a file using a shared {@link Arena}. The entire file is
 * mapped as a single {@link MemorySegment} (no 2GB limit). Multiple threads can safely read from
 * independent {@link MappedSegmentInput} instances sharing the same mapped segment.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * try (var supplier = MappedSegmentInputSupplier.open(path)) {
 *     try (var reader = supplier.open()) {
 *         reader.seek(0);
 *         int magic = reader.readInt();
 *     }
 * }
 * }</pre>
 */
public final class MappedSegmentInputSupplier implements InputSupplier {

  private final Arena arena;
  private final MemorySegment mapped;
  private final long fileSize;

  private MappedSegmentInputSupplier(Arena arena, MemorySegment mapped, long fileSize) {
    this.arena = arena;
    this.mapped = mapped;
    this.fileSize = fileSize;
  }

  /**
   * Opens and memory-maps a file with the default madvise strategy ({@link MadviseStrategy#RANDOM})
   * .
   *
   * @param path the file to map
   * @return a new supplier
   * @throws IOException if the file cannot be opened or mapped
   */
  public static MappedSegmentInputSupplier open(Path path) throws IOException {
    return open(path, MadviseStrategy.RANDOM);
  }

  /**
   * Opens and memory-maps a file with the given madvise strategy.
   *
   * @param path the file to map
   * @param strategy the madvise strategy to apply
   * @return a new supplier
   * @throws IOException if the file cannot be opened or mapped
   */
  public static MappedSegmentInputSupplier open(Path path, MadviseStrategy strategy)
      throws IOException {
    Arena arena = Arena.ofShared();
    try {
      FileChannel channel = FileChannel.open(path, StandardOpenOption.READ);
      long size = channel.size();
      MemorySegment mapped;
      if (size == 0) {
        // Cannot map zero-length file; use a zero-length slice of a trivial segment
        mapped = MemorySegment.NULL;
      } else {
        mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, size, arena);
      }
      channel.close();
      MadviseUtil.apply(mapped, strategy);
      return new MappedSegmentInputSupplier(arena, mapped, size);
    } catch (Exception e) {
      arena.close();
      throw e instanceof IOException ioe ? ioe : new IOException(e);
    }
  }

  /**
   * Creates a supplier wrapping an existing MemorySegment and Arena. The caller transfers ownership
   * of the arena to this supplier.
   */
  public static MappedSegmentInputSupplier wrap(Arena arena, MemorySegment segment) {
    return new MappedSegmentInputSupplier(arena, segment, segment.byteSize());
  }

  @Override
  public MappedSegmentInput open() {
    return new MappedSegmentInput(mapped);
  }

  @Override
  public long length() {
    return fileSize;
  }

  /** Returns the underlying mapped segment for direct access. */
  public MemorySegment segment() {
    return mapped;
  }

  @Override
  public void close() {
    arena.close();
  }
}
