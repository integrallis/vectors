# Copyright 2025-2026 Integrallis Software, LLC. Apache-2.0.
#
# End-to-end validation of the in-process jpype bridge (module_jpype.VectorsJpype).
# Runs inside the Dockerfile.jpype-validate image (which has jpype + a JDK); the
# adapter and the vectors-db classpath are mounted in. Builds HNSW and VAMANA
# in-process and checks recall@10 vs brute-force ground truth — exercising the
# real jpype Python<->JVM marshalling (float[] in, Java VectorCollection calls,
# SearchResult hit ids out).
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("VECTORS_APP_DIR", "/work"))
from module_jpype import VectorsJpype  # noqa: E402


def main():
    rng = np.random.default_rng(7)
    n, dim, nq, k = 1500, 24, 40, 10
    data = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((nq, dim)).astype(np.float32)

    gt = []
    for q in queries:
        d = np.linalg.norm(data - q, axis=1)
        gt.append(set(np.argsort(d)[:k].tolist()))

    # jpype starts ONE JVM for the process; the second config reuses it.
    for params in [
        {"indexType": "HNSW", "M": 16, "efConstruction": 200},
        {"indexType": "VAMANA", "R": 32, "L": 128, "alpha": 1.2},
    ]:
        algo = VectorsJpype("euclidean", params)
        try:
            algo.fit(data)
            algo.set_query_arguments(128)
            hits = sum(len(set(algo.query(q, k)) & gt[i]) for i, q in enumerate(queries))
            recall = hits / (nq * k)
            print(f"{params['indexType']}: recall@{k} = {recall:.4f} over {nq} queries")
            assert recall >= 0.90, f"{params['indexType']} recall too low: {recall:.4f}"
        finally:
            algo.done()
    print("JPYPE VALIDATION PASSED")


if __name__ == "__main__":
    main()
