# Copyright 2025-2026 Integrallis Software, LLC
# Licensed under the Apache License, Version 2.0.
#
# ANN-Benchmarks (github.com/erikbern/ann-benchmarks) adapter for the `vectors`
# JVM vector database. It drives an embedded `vectors-server` (Helidon SE) over
# localhost HTTP, which places `vectors` in the same "vector-database server"
# bracket as Qdrant / Milvus / Weaviate / pgvector (the per-query localhost
# round-trip is part of the measured latency, exactly as it is for those
# systems). For a future in-process (library-bracket) comparison vs hnswlib /
# FAISS, swap the HTTP calls for a jpype bridge (see README.md, module_jpype.py).
#
# One `Vectors` constructor covers every index backend; `method_param["indexType"]`
# selects HNSW / VAMANA / IVF_FLAT / IVF_PQ and the rest of the dict carries that
# backend's build knobs. For graph indexes (HNSW/VAMANA) the recall-vs-QPS curve
# is swept at query time via efSearch; for IVF, nprobe is a build-time setting in
# `vectors`, so it is swept by rebuilding (arg-groups) instead.
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

try:
    from ann_benchmarks.algorithms.base.module import BaseANN
except ImportError:  # standalone (smoke_test.py) — minimal stand-in.

    class BaseANN:
        def done(self):
            pass


_METRIC = {"euclidean": "EUCLIDEAN", "angular": "COSINE"}
_GRAPH_INDEXES = {"HNSW", "VAMANA"}

_SERVER_CMD = os.environ.get("VECTORS_SERVER_CMD", "vectors-server")
_PORT = int(os.environ.get("VECTORS_PORT", "8287"))
_DATA_DIR = os.environ.get("VECTORS_DATA_DIR", "/tmp/ann-vectors-data")
_COLLECTION = "annb"
_INGEST_BATCH = 2000


class Vectors(BaseANN):
    """ANN-Benchmarks wrapper for `vectors`, served over HTTP. Backend-agnostic."""

    def __init__(self, metric, method_param):
        if metric not in _METRIC:
            raise NotImplementedError(f"vectors: unsupported metric {metric!r}")
        self._metric = _METRIC[metric]
        self._p = dict(method_param)
        self._index = str(self._p.get("indexType", "HNSW")).upper()
        # Build threads default to the server's deterministic single-threaded build.
        # Parallel HNSW build (hnswBuildThreads > 1) currently has a self-loop bug in
        # ConcurrentHnswGraphBuilder, so only opt in via an explicit "buildThreads"
        # once that is fixed; build time is reported separately from QPS anyway.
        bt = self._p.get("buildThreads")
        self._build_threads = int(bt) if bt else None
        self._ef_search = None
        self._base = f"http://127.0.0.1:{_PORT}"
        self._proc = None
        self.name = f"vectors({self._index}, {self._build_label()})"
        self._start_server()

    # ---- server lifecycle ----

    def _start_server(self):
        os.makedirs(_DATA_DIR, exist_ok=True)
        self._proc = subprocess.Popen(
            [_SERVER_CMD, "--port", str(_PORT), "--data-dir", _DATA_DIR],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 180
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

    # ---- collection body per index backend ----

    def _build_label(self):
        p = self._p
        if self._index == "HNSW":
            return f"M={p.get('M')}, efC={p.get('efConstruction')}"
        if self._index == "VAMANA":
            return f"R={p.get('R')}, L={p.get('L')}, alpha={p.get('alpha')}"
        if self._index in ("IVF_FLAT", "IVF_PQ"):
            return f"nlist={p.get('nlist')}, nprobe={p.get('nprobe')}"
        return ""

    def _create_body(self, dim):
        p = self._p
        body = {
            "name": _COLLECTION,
            "dimension": dim,
            "metric": self._metric,
            "indexType": self._index,
        }
        if self._index == "HNSW":
            _put(body, "hnswM", p.get("M"))
            _put(body, "hnswEfConstruction", p.get("efConstruction"))
            _put(body, "hnswBuildThreads", self._build_threads)
        elif self._index == "VAMANA":
            _put(body, "vamanaMaxDegree", p.get("R"))
            _put(body, "vamanaSearchListSize", p.get("L"))
            _put(body, "vamanaAlpha", p.get("alpha"))
            _put(body, "vamanaBuildThreads", self._build_threads)
        elif self._index in ("IVF_FLAT", "IVF_PQ"):
            _put(body, "ivfK", p.get("nlist"))
            _put(body, "ivfNprobe", p.get("nprobe"))
            _put(body, "ivfMaxIter", p.get("maxIter"))
            if self._index == "IVF_PQ":
                _put(body, "ivfPqSubspaces", p.get("pqSubspaces"))
                _put(body, "ivfPqClusters", p.get("pqClusters"))
                _put(body, "ivfRescoreFactor", p.get("rescore"))
        return body

    # ---- ANN-Benchmarks API ----

    def fit(self, X):
        n = len(X)
        dim = len(X[0])
        try:
            urllib.request.urlopen(
                urllib.request.Request(self._base + "/v1/collections/" + _COLLECTION, method="DELETE"),
                timeout=30,
            )
        except urllib.error.HTTPError:
            pass
        self._post("/v1/collections", self._create_body(dim))
        for start in range(0, n, _INGEST_BATCH):
            docs = [
                {"id": str(i), "vector": _to_list(X[i])}
                for i in range(start, min(start + _INGEST_BATCH, n))
            ]
            self._post(f"/v1/collections/{_COLLECTION}/documents", {"documents": docs})
        self._post(f"/v1/collections/{_COLLECTION}/commit", {})

    def set_query_arguments(self, ef_search):
        # efSearch is the graph-index (HNSW/VAMANA) search-time beam. IVF sweeps
        # nprobe at build time, so the argument is ignored there.
        if self._index in _GRAPH_INDEXES and ef_search and int(ef_search) > 0:
            self._ef_search = int(ef_search)
            self.name = f"vectors({self._index}, {self._build_label()}, efSearch={self._ef_search})"
        else:
            self._ef_search = None
            self.name = f"vectors({self._index}, {self._build_label()})"

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


def _put(body, key, value):
    if value is not None:
        body[key] = value


def _to_list(v):
    tolist = getattr(v, "tolist", None)
    return tolist() if tolist is not None else list(v)
