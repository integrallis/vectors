#!/usr/bin/env bash
set -uo pipefail
export JAVA_HOME=/usr/lib/jvm/temurin-25-jdk-amd64; export PATH="$JAVA_HOME/bin:$PATH"
OUT=$HOME/out; mkdir -p $OUT/jmh3
# Phase 3: the e-means default's training cost against the k-means it replaced, same host, same tag
# (KMeansEMeansBenchmark exists only from 0.1.35; it trains both algorithms on the same data:
# n in {10k,50k,100k} x dim in {64,128} x k in {64,256,1024}). Annotated fork=1 warmup=2 iterations=3.
(cd ~/v0.1.35 && rm -rf vectors-bench/build/results/jmh && ./gradlew :vectors-bench:jmh -Pjmh.includes='KMeansEMeansBenchmark\.' -Pjmh.resultFormat=JSON --no-daemon > $OUT/jmh3/v0.1.35-KMeansEMeansBenchmark.log 2>&1; rc=$?; f=vectors-bench/build/results/jmh/results.json; [ -s $f ] && cp $f $OUT/jmh3/v0.1.35-KMeansEMeansBenchmark.json; echo "  v0.1.35 KMeansEMeansBenchmark exit=$rc json=$([ -s $f ] && echo yes || echo NO) $(date -u +%T)")
echo "VECJMH3-DONE $(date -u +%FT%TZ)"
