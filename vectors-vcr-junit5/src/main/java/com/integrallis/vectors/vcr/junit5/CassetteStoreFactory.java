package com.integrallis.vectors.vcr.junit5;

import com.integrallis.vectors.vcr.CassetteStore;
import java.nio.file.Path;

/**
 * SPI for constructing a {@link CassetteStore} from a {@link VCRTest} configuration.
 *
 * <p>Discovered via {@link java.util.ServiceLoader}. The first provider that returns a non-null
 * {@link CassetteStore} from {@link #create(Path)} wins; if no provider is registered, the
 * extension falls back to an {@link com.integrallis.vectors.vcr.ExactCassetteStore} over a {@link
 * com.integrallis.vectors.storage.backend.LocalFileStorageBackend} rooted at the configured {@code
 * dataDir}.
 */
public interface CassetteStoreFactory {

  /**
   * Creates a store for the given data directory, or returns {@code null} to defer to the next
   * provider (or the built-in default if no provider handles the request).
   *
   * @param dataDir the resolved absolute data directory
   * @return the store or {@code null}
   */
  CassetteStore create(Path dataDir);
}
