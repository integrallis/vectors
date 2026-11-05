#!/usr/bin/env bash
# Vectors BF16 benchmark on clean Hetzner instance
# Run this AFTER ssh'ing to the provisioned host
set -euo pipefail

export JAVA_HOME=/opt/jdk-25
export PATH="$JAVA_HOME/bin:$PATH"
export GRADLE_OPTS="--enable-native-access=ALL-UNNAMED"

cd /opt/vectors

# Compile and run JMH benchmarks for BFloat16Matrix
./gradlew :vectors-bench:jmh \
  -Pjmh.includes='BFloat16MatrixBenchmark' \
  -Pjmh.fork=3 \
  -Pjmh.warmup=5 \
  -Pjmh.iterations=10 \
  -Pjmh.timeOnIteration=1s \
  -Pjmh.resultFormat=JSON \
  --console=plain --no-daemon 2>&1 | tee /tmp/bf16-jmh.log

cp vectors-bench/build/results/jmh/results.json /tmp/bf16-results.json
echo "BENCHMARK_COMPLETE"
