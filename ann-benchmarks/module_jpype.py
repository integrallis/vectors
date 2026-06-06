# Copyright 2025-2026 Integrallis Software, LLC
# Licensed under the Apache License, Version 2.0.
#
# IN-PROCESS (library-bracket) ANN-Benchmarks adapter for `vectors`. Unlike
# module.py (which talks to vectors-server over HTTP and so sits in the
# "database server" bracket vs Qdrant/Milvus/pgvector), this variant loads
# vectors-db into the SAME process via jpype and calls VectorCollection directly
# — ZERO IPC — so vectors is measured in the in-process LIBRARY bracket vs
# hnswlib / FAISS / ScaNN.
#
# Requirements: jpype (`pip install JPype1`), a JDK 25 (JAVA_HOME), and the
# vectors-db runtime classpath. The classpath is taken from VECTORS_CLASSPATH
# (default the vectors-server distribution's lib/*.jar, which bundles vectors-db
# and every transitive dependency). See Dockerfile.jpype.
#
# NOTE: this file is exercised by the Docker image (which installs jpype); it
# cannot run in an environment without jpype. The VectorCollection call sequence
# below is the same one the vectors-db test suite covers; only the jpype
# Python<->JVM marshalling is environment-specific.
import glob
import os

import jpype

try:
    from ann_benchmarks.algorithms.base.module import BaseANN
except ImportError:

    class BaseANN:
        def done(self):
            pass


_METRIC = {"euclidean": "EUCLIDEAN", "angular": "COSINE"}
_GRAPH_INDEXES = {"HNSW", "VAMANA"}
_CLASSPATH = os.environ.get("VECTORS_CLASSPATH", "/opt/vectors-server/lib/*.jar")
_XMX = os.environ.get("VECTORS_XMX", "8g")


def _ensure_jvm():
    if jpype.isJVMStarted():
        return
    jars = sorted(glob.glob(_CLASSPATH))
    if not jars:
        raise RuntimeError(f"no jars matched VECTORS_CLASSPATH={_CLASSPATH!r}")
    # vectors needs the incubator Vector API; --enable-native-access silences the
    # FFM downcall warnings from vectors-storage's mmap layer.
    jpype.startJVM(
        f"-Xms{_XMX}",
        f"-Xmx{_XMX}",
        "--add-modules=jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED",
        "-Djava.awt.headless=true",
        classpath=jars,
    )


def _jfloats(v):
    # numpy row or list -> Java float[]. tolist() keeps marshalling jpype-version
    # agnostic; build time is reported separately from QPS so the cost is fine.
    tolist = getattr(v, "tolist", None)
    seq = tolist() if tolist is not None else list(v)
    return jpype.JArray(jpype.JFloat)(seq)


class VectorsJpype(BaseANN):
    """In-process `vectors` adapter via jpype. Backend selected by indexType."""

    def __init__(self, metric, method_param):
        if metric not in _METRIC:
            raise NotImplementedError(f"vectors: unsupported metric {metric!r}")
        _ensure_jvm()
        from com.integrallis.vectors.core import Document, SimilarityFunction
        from com.integrallis.vectors.db import IndexType, SearchRequest, VectorCollection

        self._VectorCollection = VectorCollection
        self._IndexType = IndexType
        self._SearchRequest = SearchRequest
        self._Document = Document
        self._metric = getattr(SimilarityFunction, _METRIC[metric])
        self._p = dict(method_param)
        self._index = str(self._p.get("indexType", "HNSW")).upper()
        self._ef_search = None
        self._col = None
        self.name = f"vectors-jpype({self._index})"

    def fit(self, X):
        p = self._p
        builder = (
            self._VectorCollection.builder()
            .dimension(int(len(X[0])))
            .metric(self._metric)
            .indexType(getattr(self._IndexType, self._index))
        )
        if self._index == "HNSW":
            _apply(builder.hnswM, p.get("M"), int)
            _apply(builder.hnswEfConstruction, p.get("efConstruction"), int)
        elif self._index == "VAMANA":
            _apply(builder.vamanaMaxDegree, p.get("R"), int)
            _apply(builder.vamanaSearchListSize, p.get("L"), int)
            _apply(builder.vamanaAlpha, p.get("alpha"), float)
        elif self._index in ("IVF_FLAT", "IVF_PQ"):
            _apply(builder.ivfK, p.get("nlist"), int)
            _apply(builder.ivfNprobe, p.get("nprobe"), int)
            if self._index == "IVF_PQ":
                _apply(builder.ivfPqSubspaces, p.get("pqSubspaces"), int)
                _apply(builder.ivfPqClusters, p.get("pqClusters"), int)
                _apply(builder.ivfRescoreFactor, p.get("rescore"), int)
        self._col = builder.build()
        document_of = self._Document.of
        add = self._col.add
        for i in range(len(X)):
            add(document_of(str(i), _jfloats(X[i])))
        self._col.commit()

    def set_query_arguments(self, ef_search):
        if self._index in _GRAPH_INDEXES and ef_search and int(ef_search) > 0:
            self._ef_search = int(ef_search)
            self.name = f"vectors-jpype({self._index}, efSearch={self._ef_search})"
        else:
            self._ef_search = None
            self.name = f"vectors-jpype({self._index})"

    def query(self, v, n):
        rb = self._SearchRequest.builder(_jfloats(v), int(n))
        if self._ef_search is not None:
            rb.searchListSize(self._ef_search)
        hits = self._col.search(rb.build()).hits()
        return [int(str(hits.get(i).id())) for i in range(hits.size())]

    def __str__(self):
        return self.name

    def done(self):
        if self._col is not None:
            self._col.close()
            self._col = None


def _apply(method, value, cast):
    if value is not None:
        method(cast(value))
