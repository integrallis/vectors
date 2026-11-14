| # | point | r1 forks | r2 forks | median | vs v0.1.34 | step | subject |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | v0.1.34 | 23.35, 24.60 | 24.59, 22.96 | 23.97 | 1.000 | +0.0% | Merge pull request #123 from integrallis/release/0.1.34 |
| 1 | 871dafcc | 23.58, 23.70 | 24.15, 24.45 | 23.93 | 0.998 | -0.2% | perf(hnsw): eliminate per-search allocation in pickSeeds |
| 2 | ef814631 | 23.38, 22.90 | 24.48, 24.12 | 23.75 | 0.991 | -0.7% | perf(hnsw): reuse NeighborSelector scratch buffers on pruning path |
| 3 | 686864b4 | 23.47, 24.43 | 24.37, 22.92 | 23.92 | 0.998 | +0.7% | perf(builder): generation-stamped int[] visited set in concurrent buil |
| 4 | 73bcc1b8 | 23.92, 24.11 | 23.12, 23.88 | 23.90 | 0.997 | -0.1% | perf(searcher): size scratch heaps by graph degree instead of fixed 25 |
| 5 | 97579da4 | 24.32, 24.28 | 23.72, 24.14 | 24.21 | 1.010 | +1.3% | perf(neighbor): linear-scan insertion for small NeighborArray (capacit |
| 6 | a5fbaef4 | 23.32, 25.32 | 22.68, 22.99 | 23.16 | 0.966 | -4.4% | perf(adapter): remove unnecessary defensive array clones on search pat |
| 7 | d43220ee | 23.15, 24.68 | 23.94, 24.55 | 24.25 | 1.011 | +4.7% | perf(builder): reuse search-result arrays in concurrent graph builders |
| 8 | e128b7b5 | 22.97, 23.28 | 24.87, 23.40 | 23.34 | 0.974 | -3.7% | perf(search): cache ACORN pre-filter flag in Generation |
| 9 | d79e41be | 24.67, 23.37 | 23.01, 22.82 | 23.19 | 0.968 | -0.6% | perf(searcher): size rescore heap to 256 to avoid mid-rescore growth |
| 10 | 1b82abfd | 24.12, 24.03 | 24.19, 24.59 | 24.16 | 1.008 | +4.1% | perf(hnsw): generation-stamped int[] visited set + reusable result dra |
| 11 | d54794d2 | 24.03, 24.62 | 24.77, 23.65 | 24.32 | 1.015 | +0.7% | perf(hnsw): generation-stamped dedup for searchMultiStart merge |
| 12 | 07d02399 | 23.03, 24.62 | 24.63, 24.90 | 24.62 | 1.027 | +1.2% | perf(cache): eliminate varargs Object[] allocations in FilterHasher |
| 13 | e7fe344e | 23.13, 23.78 | 24.94, 22.26 | 23.46 | 0.978 | -4.7% | perf(hnsw): cap multi-start worker creation at availableProcessors |
| 14 | 0ba0417c | 24.12, 24.55 | 24.69, 24.42 | 24.49 | 1.021 | +4.4% | perf(cache): zero-allocation QvCache lookup on cache hit |
| 15 | dda9381a | 24.67, 23.47 | 24.25, 23.41 | 23.86 | 0.995 | -2.6% | perf(db): reuse thread-local scratch for cosine query normalization |
| 16 | 8b2d0b51 | 24.23, 24.86 | 23.74, 23.38 | 23.99 | 1.001 | +0.5% | perf(db): skip BitSet.clone() on append-only commits (#167) |
| 17 | 25a38733 | 23.73, 23.54 | 22.80, 23.47 | 23.50 | 0.980 | -2.0% | perf(db): reuse stored Document on search when projection is identical |
| 18 | ca2de0d9 | 24.57, 23.67 | 25.15, 24.02 | 24.30 | 1.014 | +3.4% | perf(db): normalize vectors outside the write lock on ingest |
| 19 | dc0cff0e | 22.84, 23.06 | 24.71, 23.75 | 23.41 | 0.976 | -3.7% | perf(db): eliminate List.copyOf allocation in SearchResult constructor |
| 20 | 29a28e0f | 23.33, 23.01 | 24.92, 23.53 | 23.43 | 0.978 | +0.1% | perf(db): eliminate double vector clone on ingest |
| 21 | b500e86e | 22.32, 24.40 | 24.56, 23.09 | 23.75 | 0.991 | +1.3% | perf(hnsw): eliminate per-search default-scorer allocation (#177) |
| 22 | b575aad0 | 24.35, 24.54 | 23.34, 22.64 | 23.85 | 0.995 | +0.4% | Merge pull request #181 from integrallis/docs/score-scale-explicit |
| 23 | 4b22ff0c | 24.84, 22.86 | 23.38, 24.36 | 23.87 | 0.996 | +0.1% | feat: expose raw scores via SearchResult.rawScoreOf() and SimilarityFu |
| 24 | v0.1.35 | 22.92, 24.54 | 23.75, 23.30 | 23.52 | 0.981 | -1.5% | docs: bump version pills and antora config to 0.1.35 |
