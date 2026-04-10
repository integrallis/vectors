/**
 * Dataset loaders and path registry for ANN-benchmark data formats.
 *
 * <p>Supported formats:
 *
 * <ul>
 *   <li>{@link com.integrallis.vectors.bench.dataset.FvecsLoader} — TexMex/BIGANN binary formats
 *       ({@code .fvecs}, {@code .ivecs}, {@code .bvecs}).
 *   <li>{@link com.integrallis.vectors.bench.dataset.Hdf5Loader} — ANN-Benchmarks HDF5 format
 *       (GloVe, NYTimes, Deep, etc.).
 *   <li>{@link com.integrallis.vectors.bench.dataset.DatasetRegistry} — runtime path resolution via
 *       the {@code VECTORS_BENCH_DATA} environment variable with bundled-dataset fallbacks.
 * </ul>
 */
package com.integrallis.vectors.bench.dataset;
