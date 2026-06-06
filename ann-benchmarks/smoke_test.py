# Copyright 2025-2026 Integrallis Software, LLC
# Licensed under the Apache License, Version 2.0.
#
# Standalone verification of the ANN-Benchmarks adapter WITHOUT the full harness:
# boots vectors-server via the locally-built distribution, drives the `Vectors`
# wrapper through fit()/set_query_arguments()/query() on a tiny random dataset,
# and asserts the HNSW recall@10 vs brute-force ground truth is high.
#
# Run:
#   ./gradlew :vectors-server:installDist
#   VECTORS_SERVER_CMD=$PWD/vectors-server/build/install/vectors-server/bin/vectors-server \
#     python3 ann-benchmarks/smoke_test.py
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from module import Vectors  # noqa: E402


def main():
    rng = np.random.default_rng(42)
    n, dim, nq, k = 2000, 32, 50, 10
    data = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((nq, dim)).astype(np.float32)

    # Brute-force Euclidean ground truth.
    gt = []
    for q in queries:
        d = np.linalg.norm(data - q, axis=1)
        gt.append(set(np.argsort(d)[:k].tolist()))

    os.environ.setdefault("VECTORS_DATA_DIR", tempfile.mkdtemp(prefix="annb-smoke-"))
    algo = Vectors("euclidean", {"M": 16, "efConstruction": 200})
    try:
        algo.fit(data)
        algo.set_query_arguments(128)
        hits = 0
        for i, q in enumerate(queries):
            got = set(algo.query(q, k))
            hits += len(got & gt[i])
        recall = hits / (nq * k)
        print(f"recall@{k} = {recall:.4f} over {nq} queries (efSearch=128)")
        assert recall >= 0.90, f"recall too low: {recall:.4f}"
        print("SMOKE TEST PASSED")
    finally:
        algo.done()


if __name__ == "__main__":
    main()
