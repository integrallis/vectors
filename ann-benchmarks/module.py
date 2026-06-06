# Copyright 2025-2026 Integrallis Software, LLC
# Licensed under the Apache License, Version 2.0.
#
# ANN-Benchmarks (github.com/erikbern/ann-benchmarks) adapter for the `vectors`
# JVM vector database. It drives an embedded `vectors-server` (Helidon SE) over
# localhost HTTP, which places `vectors` in the same "vector-database server"
# bracket as Qdrant / Milvus / Weaviate / pgvector (the per-query localhost
# round-trip is part of the measured latency, exactly as it is for those
# systems). For a future in-process (library-bracket) comparison vs hnswlib /
# FAISS, swap the HTTP calls for a jpype bridge — see README.md.
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

try:
    # Provided by the ann-benchmarks harness at runtime.
    from ann_benchmarks.algorithms.base.module import BaseANN
except ImportError:  # standalone (smoke_test.py) — minimal stand-in.

    class BaseANN:
        def done(self):
            pass


# ann-benchmarks metric name -> vectors SimilarityFunction.
_METRIC = {"euclidean": "EUCLIDEAN", "angular": "COSINE"}

# Launcher for the vectors-server distribution (built via
# `./gradlew :vectors-server:installDist`). The Dockerfile sets this to the
# installed path; locally, smoke_test.py points it at build/install.
_SERVER_CMD = os.environ.get("VECTORS_SERVER_CMD", "vectors-server")
_PORT = int(os.environ.get("VECTORS_PORT", "8287"))
_DATA_DIR = os.environ.get("VECTORS_DATA_DIR", "/tmp/ann-vectors-data")
_COLLECTION = "annb"
_INGEST_BATCH = 2000


class Vectors(BaseANN):
    """ANN-Benchmarks wrapper for the `vectors` HNSW index, served over HTTP."""

    def __init__(self, metric, method_param):
        if metric not in _METRIC:
            raise NotImplementedError(f"vectors: unsupported metric {metric!r}")
        self._metric = _METRIC[metric]
        self._m = int(method_param["M"])
        self._ef_construction = int(method_param["efConstruction"])
        self._ef_search = None
        self._base = f"http://127.0.0.1:{_PORT}"
        self._dim = None
        self._proc = None
        self.name = f"vectors(M={self._m}, efC={self._ef_construction})"
        self._start_server()

    # ---- server lifecycle ----

    def _start_server(self):
        os.makedirs(_DATA_DIR, exist_ok=True)
        self._proc = subprocess.Popen(
            [_SERVER_CMD, "--port", str(_PORT), "--data-dir", _DATA_DIR],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 180  # JVM start + incubator module load
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("vectors-server exited during startup")
            try:
                with urllib.request.urlopen(self._base + "/v1/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        raise TimeoutError("vectors-server did not become healthy within 180s")

    # ---- HTTP helpers ----

    def _post(self, path, body):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self._base + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read()
            return json.loads(raw) if raw else None

    # ---- ANN-Benchmarks API ----

    def fit(self, X):
        n = len(X)
        self._dim = len(X[0])
        try:
            urllib.request.urlopen(
                urllib.request.Request(self._base + "/v1/collections/" + _COLLECTION, method="DELETE"),
                timeout=30,
            )
        except urllib.error.HTTPError:
            pass  # 404 if absent — fine
        self._post(
            "/v1/collections",
            {
                "name": _COLLECTION,
                "dimension": self._dim,
                "metric": self._metric,
                "indexType": "HNSW",
                "hnswM": self._m,
                "hnswEfConstruction": self._ef_construction,
            },
        )
        for start in range(0, n, _INGEST_BATCH):
            docs = [
                {"id": str(i), "vector": _to_list(X[i])}
                for i in range(start, min(start + _INGEST_BATCH, n))
            ]
            self._post(f"/v1/collections/{_COLLECTION}/documents", {"documents": docs})
        self._post(f"/v1/collections/{_COLLECTION}/commit", {})

    def set_query_arguments(self, ef_search):
        self._ef_search = int(ef_search)
        self.name = (
            f"vectors(M={self._m}, efC={self._ef_construction}, efSearch={self._ef_search})"
        )

    def query(self, v, n):
        body = {"queryVector": _to_list(v), "k": n}
        if self._ef_search is not None:
            body["efSearch"] = self._ef_search
        resp = self._post(f"/v1/collections/{_COLLECTION}/search", body)
        return [int(hit["id"]) for hit in resp["hits"]]

    def __str__(self):
        return self.name

    def done(self):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


def _to_list(v):
    # Accept numpy arrays (ann-benchmarks) or plain lists (smoke test).
    tolist = getattr(v, "tolist", None)
    return tolist() if tolist is not None else list(v)
